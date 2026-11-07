// baseline 부하 측정: AI 서버의 POST /trips/place-selection에 CONC건을 동시에 보내는 일을 ITER번 연속으로 반복한다.
// 가상 사용자 1명이 한 반복마다 http.batch로 CONC건을 한꺼번에 보내고, 모두 끝나면 바로 다음 반복을 시작한다.
// 반복 사이에 쉬지 않아서 직전 작업 후 RunPod의 Idle timeout(콘솔 기본 5초) 안에 다음 요청이 간다.
// 첫 반복(iter 0)은 예열이라 warmup=true로 표시되고, 집계에서 뺀다. 예열이 필요 없으면 -e WARMUP=0.
//   k6 run -e CONC=1 -e ITER=6 -e TOKEN="$TOKEN" place_selection.js   (N=1, 예열 1 + 측정 5)
//   k6 run -e CONC=2 -e ITER=6 -e TOKEN="$TOKEN" place_selection.js   (N=2, 예열 1 + 측정 5)
// cold 측정은 워커가 내려간(Worker states Idle) 것을 확인한 뒤 -e ITER=1 -e WARMUP=0 으로 1회 실행한다.
// 줄 세우기(place_selection_queue_capacity, 기본 9) 확인: -e CONC=12 -e ITER=2 이면 반복마다 9건은 줄을 서서 순서대로 200(k번째 약 2.7 × k초), 3건은 바로 503이 나와야 한다(호출 약 36개).
import http from 'k6/http';
import exec from 'k6/execution';
import { check } from 'k6';

const CONC = parseInt(__ENV.CONC || '1');
const ITER = parseInt(__ENV.ITER || '6');
const WARMUP = (__ENV.WARMUP || '1') === '1';
const DAYS = parseInt(__ENV.DAYS || '2'); // 시작일 + DAYS일. 2이면 3일 일정
const BASE_URL = __ENV.BASE_URL || 'http://localhost:8000';
const members = JSON.parse(open('./members.json'));

export const options = {
  // http.batch는 같은 호스트로 동시에 보내는 요청을 기본 6개로 제한한다(batchPerHost 기본값). CONC건을 모두 동시에 보내려고 CONC로 올린다
  batch: CONC,
  batchPerHost: CONC,
  scenarios: {
    repeat: { executor: 'per-vu-iterations', vus: 1, iterations: ITER, maxDuration: '60m' },
  },
};

// 서비스 날짜는 한국 시간 기준이다
function kstDate(offsetDays) {
  return new Date(Date.now() + 9 * 3600 * 1000 + offsetDays * 86400 * 1000).toISOString().slice(0, 10);
}

export default function () {
  const iter = exec.scenario.iterationInTest;
  const body = JSON.stringify({
    region: '서울',
    start_date: kstDate(0),
    end_date: kstDate(DAYS),
    members: members,
  });
  const params = {
    headers: { 'Content-Type': 'application/json', 'X-Internal-Token': __ENV.TOKEN },
    timeout: '130s', // k6 기본값 60초는 baseline의 순차 대기(최악 약 110초)보다 짧다
  };
  const requests = [];
  for (let i = 0; i < CONC; i++) {
    requests.push(['POST', `${BASE_URL}/trips/place-selection`, body, params]);
  }
  const startedAt = new Date().toISOString();
  const responses = http.batch(requests);
  const endedAt = new Date().toISOString();
  responses.forEach((res, i) => {
    check(res, { 'status 200': (r) => r.status === 200 });
    console.log(JSON.stringify({
      iter: iter, req: i, warmup: WARMUP && iter === 0, conc: CONC,
      status: res.status, seconds: Number((res.timings.duration / 1000).toFixed(2)),
      startedAt: startedAt, endedAt: endedAt,
    }));
  });
}
