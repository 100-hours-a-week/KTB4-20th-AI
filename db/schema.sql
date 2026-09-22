-- 1. places 테이블
CREATE TABLE places (
    id INT AUTO_INCREMENT PRIMARY KEY,
    google_place_id VARCHAR(100) UNIQUE NOT NULL,
    name VARCHAR(200),
    rating FLOAT,
    user_rating_count INT,
    editorial_summary TEXT,
    latitude DECIMAL(10,7),
    longitude DECIMAL(10,7),
    created_at DATETIME,
    updated_at DATETIME
);

-- 2. place_categories 테이블 (N:M 연결)
CREATE TABLE place_categories (
    id INT AUTO_INCREMENT PRIMARY KEY,
    place_id INT NOT NULL,
    category VARCHAR(30) NOT NULL,
    FOREIGN KEY (place_id) REFERENCES places(id),
    INDEX idx_category (category)
);

-- 3. place_types 테이블 (N:M 연결)
CREATE TABLE place_types (
    id INT AUTO_INCREMENT PRIMARY KEY,
    place_id INT NOT NULL,
    type VARCHAR(50) NOT NULL,
    FOREIGN KEY (place_id) REFERENCES places(id)
);