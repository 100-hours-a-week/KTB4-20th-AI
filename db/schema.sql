-- 1. ai_places 테이블
CREATE TABLE ai_places (
    -- 기존 컬럼 10개
    id INT AUTO_INCREMENT PRIMARY KEY,
    google_place_id VARCHAR(100) UNIQUE NOT NULL,
    name VARCHAR(200),
    rating FLOAT,
    user_rating_count INT,
    editorial_summary TEXT,
    latitude DECIMAL(10,7),
    longitude DECIMAL(10,7),
    created_at DATETIME,
    updated_at DATETIME,

    -- scalar 컬럼 40개
    resource_name VARCHAR(300),
    adr_format_address TEXT,
    business_status VARCHAR(30),
    formatted_address VARCHAR(500),
    short_formatted_address VARCHAR(300),
    google_maps_uri VARCHAR(500),
    icon_background_color VARCHAR(20),
    icon_mask_base_uri VARCHAR(500),
    moved_place VARCHAR(300),
    moved_place_id VARCHAR(255),
    primary_type VARCHAR(50),
    pure_service_area_business BOOLEAN,
    utc_offset_minutes INT,
    international_phone_number VARCHAR(30),
    national_phone_number VARCHAR(30),
    price_level VARCHAR(30),
    website_uri TEXT,
    allows_dogs BOOLEAN, curbside_pickup BOOLEAN, delivery BOOLEAN,
    dine_in BOOLEAN, good_for_children BOOLEAN, good_for_groups BOOLEAN,
    good_for_watching_sports BOOLEAN, live_music BOOLEAN, menu_for_children BOOLEAN,
    outdoor_seating BOOLEAN, reservable BOOLEAN, restroom BOOLEAN,
    serves_beer BOOLEAN, serves_breakfast BOOLEAN, serves_brunch BOOLEAN,
    serves_cocktails BOOLEAN, serves_coffee BOOLEAN, serves_dessert BOOLEAN,
    serves_dinner BOOLEAN, serves_lunch BOOLEAN, serves_vegetarian_food BOOLEAN,
    serves_wine BOOLEAN, takeout BOOLEAN,

    -- 펼친 컬럼 30개
    primary_type_display_name VARCHAR(100),
    google_maps_type_label VARCHAR(100),
    wheelchair_accessible_parking BOOLEAN, wheelchair_accessible_entrance BOOLEAN,
    wheelchair_accessible_restroom BOOLEAN, wheelchair_accessible_seating BOOLEAN,
    parking_free_lot BOOLEAN, parking_paid_lot BOOLEAN,
    parking_free_street BOOLEAN, parking_paid_street BOOLEAN,
    parking_valet BOOLEAN, parking_free_garage BOOLEAN, parking_paid_garage BOOLEAN,
    payment_accepts_credit_cards BOOLEAN, payment_accepts_debit_cards BOOLEAN,
    payment_accepts_cash_only BOOLEAN, payment_accepts_nfc BOOLEAN,
    plus_code_global VARCHAR(30),
    plus_code_compound VARCHAR(100),
    time_zone_id VARCHAR(50),
    opening_date DATE,
    google_maps_directions_uri VARCHAR(500),
    google_maps_place_uri VARCHAR(500),
    google_maps_write_review_uri VARCHAR(500),
    google_maps_reviews_uri VARCHAR(500),
    google_maps_photos_uri VARCHAR(500),
    viewport_low_latitude DECIMAL(10,7), viewport_low_longitude DECIMAL(10,7),
    viewport_high_latitude DECIMAL(10,7), viewport_high_longitude DECIMAL(10,7),

    -- JSON 컬럼 21개
    address_components JSON, address_descriptor JSON,
    attributions JSON, consumer_alert JSON,
    containing_places JSON, sub_destinations JSON,
    postal_address JSON, entrances JSON,
    navigation_points JSON, transit_station JSON,
    price_range JSON, regular_opening_hours JSON,
    current_opening_hours JSON, regular_secondary_opening_hours JSON,
    current_secondary_opening_hours JSON, generative_summary JSON,
    review_summary JSON, neighborhood_summary JSON,
    ev_charge_amenity_summary JSON, ev_charge_options JSON,
    fuel_options JSON
);

-- 2. ai_place_categories 테이블 (N:M 연결)
CREATE TABLE ai_place_categories (
    id INT AUTO_INCREMENT PRIMARY KEY,
    place_id INT NOT NULL,
    category VARCHAR(30) NOT NULL,
    FOREIGN KEY (place_id) REFERENCES ai_places(id),
    UNIQUE KEY uq_place_category (place_id, category),
    INDEX idx_category (category)
);

-- 3. ai_place_types 테이블 (N:M 연결)
CREATE TABLE ai_place_types (
    id INT AUTO_INCREMENT PRIMARY KEY,
    place_id INT NOT NULL,
    type VARCHAR(50) NOT NULL,
    FOREIGN KEY (place_id) REFERENCES ai_places(id)
);

-- 4. ai_place_regions 테이블
CREATE TABLE ai_place_regions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    place_id INT NOT NULL,
    region VARCHAR(30) NOT NULL,
    FOREIGN KEY (place_id) REFERENCES ai_places(id),
    UNIQUE KEY uq_place_region (place_id, region),
    INDEX idx_region (region)
);

CREATE TABLE ai_place_reviews (
    id INT AUTO_INCREMENT PRIMARY KEY,
    place_id INT NOT NULL,
    review_name VARCHAR(500),
    relative_publish_time_description VARCHAR(50),
    rating FLOAT,
    review_text TEXT,
    original_text TEXT,
    author_display_name VARCHAR(100),
    author_uri VARCHAR(500),
    author_photo_uri VARCHAR(500),
    publish_time DATETIME,
    flag_content_uri VARCHAR(500),
    google_maps_uri VARCHAR(500),
    FOREIGN KEY (place_id) REFERENCES ai_places(id),
    INDEX idx_review_place (place_id)
);