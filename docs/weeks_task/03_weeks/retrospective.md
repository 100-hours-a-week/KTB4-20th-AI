# retrospective

## 260920

### - Pydantic Generic Response 설계

**문제 상황**: trips 엔드포인트(place-selection, precheck, course-recommendation)마다 응답 형태가 다른데, 매번 새 응답 클래스를 처음부터 만들면 `status_code`+`data`라는 공통 구조가 중복됨. 이걸 재사용 가능한 방식으로 설계해야 했음.

**부족한 개념**: Pydantic에서 여러 엔드포인트가 공통 틀(상태 코드+데이터)을 공유하면서도, 각자 다른 데이터 타입을 담으려면 상속과 제네릭(Generic) 중 무엇을 써야 하는지 몰랐음.

**알게 된 사실**:

- 상속(`class A_Data(Response_Data)`) 방식은 `Response.data` 필드가 부모 타입으로 선언되어, 실제 자식 타입 전용 필드에 접근할 때 타입 체커가 못 알아챔.
- `Generic[T]`를 쓰면 `Response[PlaceSelectionResponseData]`처럼 사용 시점에 실제 타입을 지정할 수 있어, `data.places`처럼 자식 전용 필드에 안전하게 접근 가능함.
- 공통 필드·로직이 하나도 없는 부모 클래스(`Request`)는 상속해도 `BaseModel`을 직접 상속하는 것과 동작이 완전히 같아, 존재 자체가 무의미함 — 필요해질 때 다시 만드는 게 맞음(YAGNI).

**개념이 포함된 섹션**: PL(Programming Language) — 타입 시스템(제네릭, 상속)
