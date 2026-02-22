# Mandi Price API & Enhanced Renting Module Implementation Plan

## Overview

This plan outlines the implementation of two major modules for the Kalpataru AgriTech platform:

1. **Mandi Price API Module** - Integration with Government of India's data.gov.in APIs for real-time commodity prices
2. **Enhanced Renting Module** - Expansion of existing rental system to include Tractors, Drones, Seeds, and agricultural equipment

---

## Module 1: Mandi Price API Integration

### 1.1 API Endpoints to Integrate

The Government of India provides two key APIs:

#### API 1: Current Daily Price of Various Commodities
- **Endpoint**: `https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070`
- **Purpose**: Get current daily prices from various markets
- **Filters Available**:
  - `filters[state.keyword]` - State filter
  - `filters[district]` - District filter
  - `filters[market]` - Market filter
  - `filters[commodity]` - Commodity filter
  - `filters[variety]` - Variety filter
  - `filters[grade]` - Grade filter

#### API 2: Variety-wise Daily Market Prices
- **Endpoint**: `https://api.data.gov.in/resource/35985678-0d79-46b4-9ed6-6f13308a1d24`
- **Purpose**: Get variety-wise daily market prices
- **Filters Available**:
  - `filters[State]` - State filter
  - `filters[District]` - District filter
  - `filters[Commodity]` - Commodity filter
  - `filters[Arrival_Date]` - Arrival Date filter

### 1.2 Backend Architecture

```
backend/
├── services/
│   └── mandi_service.py          # API client for data.gov.in
├── models/
│   └── mandi_price.py            # Cached price data models
├── api/v1/
│   └── mandi.py                  # REST API endpoints
├── schemas/
│   └── mandi_schema.py           # Request/Response schemas
└── tasks/
    └── mandi_tasks.py            # Celery tasks for scheduled sync
```

### 1.3 Data Models

```python
# MandiPrice Model
class MandiPrice:
    id: int
    state: str
    district: str
    market: str
    commodity: str
    variety: str
    grade: str
    arrival_date: date
    min_price: float
    max_price: float
    modal_price: float      # Most common price
    commodity_code: str
    created_at: datetime
    updated_at: datetime

# MandiSyncLog Model
class MandiSyncLog:
    id: int
    sync_type: str          # 'daily' or 'variety'
    records_fetched: int
    sync_status: str
    error_message: str
    started_at: datetime
    completed_at: datetime
```

### 1.4 API Endpoints Design

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/mandi/prices` | Get current prices with filters |
| GET | `/api/v1/mandi/prices/<commodity>` | Get prices for specific commodity |
| GET | `/api/v1/mandi/variety-prices` | Get variety-wise prices |
| GET | `/api/v1/mandi/states` | List available states |
| GET | `/api/v1/mandi/districts` | List districts by state |
| GET | `/api/v1/mandi/markets` | List markets by district |
| GET | `/api/v1/mandi/commodities` | List all available commodities |
| POST | `/api/v1/mandi/sync` | Trigger manual sync (admin) |
| GET | `/api/v1/mandi/trends/<commodity>` | Price trends analysis |

### 1.5 Service Layer

```python
# mandi_service.py
class MandiService:
    BASE_URL = "https://api.data.gov.in/resource/"
    API_KEY = os.getenv('MANDI_API_KEY')
    
    @staticmethod
    def get_daily_prices(state=None, district=None, market=None, 
                         commodity=None, variety=None, grade=None,
                         offset=0, limit=50):
        """Fetch current daily prices from API 1"""
        
    @staticmethod
    def get_variety_prices(state=None, district=None, 
                           commodity=None, arrival_date=None):
        """Fetch variety-wise prices from API 2"""
        
    @staticmethod
    def get_price_trends(commodity, district, days=30):
        """Analyze price trends over time"""
        
    @staticmethod
    def sync_prices():
        """Scheduled task to sync and cache prices"""
```

### 1.6 Caching Strategy

- Use Redis cache for frequently accessed price data
- Cache TTL: 6 hours for current prices
- Background sync every 4 hours via Celery
- Fallback to cached data if API is unavailable

---

## Module 2: Enhanced Renting Module

### 2.1 Current State Analysis

The existing rental system includes:
- [`Equipment`](AgriTech/backend/models/equipment.py:5) model with basic fields
- [`RentalBooking`](AgriTech/backend/models/equipment.py:57) model with status workflow
- [`RentalService`](AgriTech/backend/services/rental_service.py:8) with booking logic
- [`equipment_bp`](AgriTech/backend/api/v1/equipment.py:6) blueprint

### 2.2 Enhanced Categories

```
Rental Categories:
├── Machinery
│   ├── Tractors (various HP ranges)
│   ├── Harvesters
│   ├── Threshers
│   ├── Tillers
│   └── Sprayers
├── Technology
│   ├── Drones (spraying, mapping, monitoring)
│   ├── Sensors (soil, weather)
│   └── GPS Equipment
├── Seeds & Planting
│   ├── Seeds (certified, hybrid)
│   ├── Seedlings
│   └── Saplings
├── Irrigation
│   ├── Pumps
│   ├── Pipes
│   └── Sprinklers
├── Tools
│   ├── Hand Tools
│   ├── Power Tools
│   └── Safety Equipment
├── Agrochemicals
│   ├── Fertilizers (organic, chemical)
│   ├── Pesticides (insecticides, fungicides, herbicides)
│   ├── Growth Promoters
│   └── Biopesticides
└── Farm Labor
    ├── Skilled Workers (machine operators, irrigation specialists)
    ├── Unskilled Workers (harvesting, planting)
    ├── Specialized Services (grafting, pruning)
    └── Equipment Operators (drone pilots, tractor operators)
```

### 2.3 Enhanced Data Models

```python
# Enhanced Equipment Model
class Equipment:
    # Existing fields...
    category: str              # Enhanced categories
    sub_category: str          # Sub-category within main category
    brand: str                 # Brand name
    model_number: str          # Model/variant
    year_of_manufacture: int   # For machinery
    condition: str             # NEW, EXCELLENT, GOOD, FAIR
    quantity_available: int    # For seeds/consumables
    
    # Rental-specific
    rental_type: str           # HOURLY, DAILY, WEEKLY, MONTHLY
    security_deposit: float
    minimum_rental_period: int
    maximum_rental_period: int
    
    # Location & Delivery
    pickup_location: str
    delivery_available: bool
    delivery_radius_km: int
    delivery_charges: float
    
    # Verification
    is_verified: bool          # Admin verified listing
    verification_date: datetime
    documents_url: str         # Ownership/quality documents
    
    # For Seeds specifically
    seed_variety: str
    germination_rate: float
    certification_number: str
    expiry_date: date

# New: Equipment Review Model
class EquipmentReview:
    id: int
    equipment_id: int
    reviewer_id: int
    booking_id: int
    rating: int                # 1-5 stars
    review_text: str
    performance_rating: int
    condition_rating: int
    value_rating: int
    created_at: datetime

# New: Equipment Availability Exception
class EquipmentAvailabilityException:
    id: int
    equipment_id: int
    exception_type: str        # MAINTENANCE, PERSONAL, HOLIDAY
    start_date: date
    end_date: date
    notes: str

# New: Agrochemical Listing (Fertilizers/Pesticides)
class AgrochemicalListing:
    id: int
    seller_id: int
    product_name: str
    category: str              # FERTILIZER, PESTICIDE, GROWTH_PROMOTER
    sub_category: str          # Organic, Chemical, Insecticide, etc.
    brand: str
    active_ingredient: str     # For pesticides
    npk_ratio: str             # For fertilizers e.g., "10-26-26"
    quantity_available: float
    unit: str                  # KG, LITER, BAG
    price_per_unit: float
    application_method: str    # SOIL, FOLIAR, DRIP
    target_crops: str          # JSON array of suitable crops
    target_pests: str          # JSON array for pesticides
    safety_interval_days: int  # Days before harvest
    expiry_date: date
    certification_number: str
    is_organic: bool
    handling_instructions: str
    safety_equipment_required: str
    image_url: str
    is_verified: bool
    created_at: datetime
    updated_at: datetime

# New: Farm Labor Listing
class FarmLaborListing:
    id: int
    worker_id: int             # Reference to User
    worker_type: str           # SKILLED, UNSKILLED, SPECIALIZED
    skills: str                # JSON array of skills
    certifications: str        # JSON array of certifications
    experience_years: int
    daily_wage: float
    hourly_wage: float
    availability_start: date
    availability_end: date
    preferred_location: str
    languages_known: str       # JSON array
    accommodation_required: bool
    minimum_engagement_days: int
    profile_photo: str
    id_proof_url: str          # Verification document
    is_verified: bool
    rating: float
    total_jobs: int
    created_at: datetime
    updated_at: datetime

# New: Labor Booking
class LaborBooking:
    id: int
    labor_listing_id: int
    farmer_id: int
    start_date: date
    end_date: date
    work_type: str             # PLANTING, HARVESTING, SPRAYING, etc.
    daily_hours: int
    total_amount: float
    status: str                # PENDING, CONFIRMED, IN_PROGRESS, COMPLETED, CANCELLED
    payment_status: str
    farmer_review: str
    worker_review: str
    farmer_rating: int
    worker_rating: int
    created_at: datetime
```

### 2.4 API Endpoints Design

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/rentals/categories` | List all rental categories |
| GET | `/api/v1/rentals/equipment` | Search equipment with filters |
| GET | `/api/v1/rentals/equipment/<id>` | Get equipment details |
| POST | `/api/v1/rentals/equipment` | List equipment for rent |
| PUT | `/api/v1/rentals/equipment/<id>` | Update equipment listing |
| DELETE | `/api/v1/rentals/equipment/<id>` | Remove equipment listing |
| GET | `/api/v1/rentals/equipment/<id>/availability` | Check availability calendar |
| POST | `/api/v1/rentals/bookings` | Create rental booking |
| GET | `/api/v1/rentals/bookings` | Get user's bookings |
| PUT | `/api/v1/rentals/bookings/<id>/status` | Update booking status |
| POST | `/api/v1/rentals/bookings/<id>/review` | Submit review after rental |
| GET | `/api/v1/rentals/equipment/<id>/reviews` | Get equipment reviews |
| GET | `/api/v1/rentals/my-equipment` | Get owner's equipment list |
| GET | `/api/v1/rentals/dashboard` | Owner dashboard stats |
| **Agrochemicals** | | |
| GET | `/api/v1/rentals/agrochemicals` | List agrochemicals with filters |
| GET | `/api/v1/rentals/agrochemicals/<id>` | Get agrochemical details |
| POST | `/api/v1/rentals/agrochemicals` | List agrochemical for sale/rent |
| GET | `/api/v1/rentals/agrochemicals/categories` | Get agrochemical categories |
| **Farm Labor** | | |
| GET | `/api/v1/rentals/labor` | List available workers |
| GET | `/api/v1/rentals/labor/<id>` | Get worker profile |
| POST | `/api/v1/rentals/labor` | Register as farm worker |
| POST | `/api/v1/rentals/labor/bookings` | Book a worker |
| GET | `/api/v1/rentals/labor/bookings` | Get labor bookings |
| PUT | `/api/v1/rentals/labor/bookings/<id>` | Update labor booking |
| GET | `/api/v1/rentals/labor/skills` | List available skills |

### 2.5 Category-Specific Features

#### Tractors
- HP range filter (under 30HP, 30-50HP, 50-75HP, above 75HP)
- 2WD/4WD specification
- Implement compatibility list
- Operator availability option
- Fuel type (Diesel/Petrol)

#### Drones
- Type: Spraying, Mapping, Monitoring
- Coverage capacity per charge
- Payload capacity
- Operator required (yes/no)
- Flight time per battery
- Certification requirements

#### Seeds
- Crop type and variety
- Quantity available (kg)
- Germination rate
- Certification details
- Expiry date
- Bulk pricing options
- Minimum order quantity

#### Fertilizers
- Type: Organic, Chemical, Bio-fertilizers
- NPK ratio specification
- Quantity available (kg/liters)
- Application method (soil, foliar, drip)
- Crop suitability list
- Organic certification (if applicable)
- Expiry date
- Bulk pricing tiers

#### Pesticides
- Type: Insecticide, Fungicide, Herbicide, Rodenticide
- Active ingredient percentage
- Target pests/diseases
- Application dosage guidelines
- Safety interval before harvest
- Quantity available (liters/kg)
- Safety equipment required
- Handling instructions

#### Farm Labor
- Worker type: Skilled, Unskilled, Specialized
- Skills/certifications (tractor operation, grafting, etc.)
- Daily/hourly wage rate
- Availability calendar
- Experience years
- Languages known
- Location preference
- Accommodation required (yes/no)
- Minimum engagement period

---

## Architecture Diagram

```mermaid
graph TB
    subgraph Frontend
        A[Mandi Price Page]
        B[Rental Marketplace Page]
        C[Equipment Detail Page]
        D[Booking Management]
    end
    
    subgraph API Gateway
        E[Flask REST API]
    end
    
    subgraph Backend Services
        F[MandiService]
        G[RentalService]
        H[BookingService]
        I[ReviewService]
    end
    
    subgraph External APIs
        J[data.gov.in API 1]
        K[data.gov.in API 2]
    end
    
    subgraph Data Layer
        L[(PostgreSQL)]
        M[(Redis Cache)]
    end
    
    subgraph Background Tasks
        N[Celery Workers]
    end
    
    A --> E
    B --> E
    C --> E
    D --> E
    
    E --> F
    E --> G
    E --> H
    E --> I
    
    F --> J
    F --> K
    F --> M
    
    G --> L
    H --> L
    I --> L
    
    N --> F
    N --> L
```

---

## Implementation Phases

### Phase 1: Mandi Price API
1. Create MandiService with API integration
2. Implement data models and caching
3. Build REST API endpoints
4. Create frontend price display page
5. Add price alert functionality

### Phase 2: Enhanced Rental Categories
1. Extend Equipment model with new fields
2. Create category-specific schemas
3. Update RentalService for new categories
4. Build category-specific frontend views
5. Implement review system

### Phase 3: Advanced Features
1. Price trend analysis and charts
2. Equipment recommendation engine
3. Delivery tracking integration
4. Payment gateway integration
5. Mobile-responsive design

---

## Configuration Requirements

### Environment Variables
```env
# Mandi API Configuration
MANDI_API_KEY=579b464db66ec23bdd0000018263afcca3cf4ce05558a68fceb1f2ec
MANDI_API_BASE_URL=https://api.data.gov.in/resource/
MANDI_CACHE_TTL=21600  # 6 hours in seconds

# Rental Configuration
MAX_RENTAL_DAYS=90
DEFAULT_SECURITY_DEPOSIT_PERCENT=20
RENTAL_CANCELLATION_HOURS=24
```

---

## Testing Strategy

### Unit Tests
- MandiService API calls with mocked responses
- RentalService booking logic
- Price calculation algorithms
- Availability conflict detection

### Integration Tests
- End-to-end booking flow
- API response validation
- Cache invalidation
- External API error handling

### Performance Tests
- Concurrent booking requests
- Large dataset pagination
- Cache hit/miss ratios

---

## Security Considerations

1. **API Key Protection**: Store Mandi API key in environment variables
2. **Rate Limiting**: Implement rate limiting on all rental endpoints
3. **Input Validation**: Sanitize all filter parameters
4. **Authorization**: Verify ownership for equipment updates
5. **Payment Security**: Use escrow for rental payments

---

## Files to Create/Modify

### New Files
- `AgriTech/backend/services/mandi_service.py`
- `AgriTech/backend/models/mandi_price.py`
- `AgriTech/backend/api/v1/mandi.py`
- `AgriTech/backend/schemas/mandi_schema.py`
- `AgriTech/backend/tasks/mandi_tasks.py`
- `AgriTech/backend/models/equipment_review.py`
- `AgriTech/backend/services/equipment_review_service.py`
- `AgriTech/backend/models/agrochemical.py`
- `AgriTech/backend/models/farm_labor.py`
- `AgriTech/backend/services/agrochemical_service.py`
- `AgriTech/backend/services/farm_labor_service.py`
- `AgriTech/backend/api/v1/agrochemicals.py`
- `AgriTech/backend/api/v1/farm_labor.py`
- `AgriTech/backend/schemas/agrochemical_schema.py`
- `AgriTech/backend/schemas/farm_labor_schema.py`
- `AgriTech/mandi_prices.html`
- `AgriTech/rental_marketplace.html`
- `AgriTech/agrochemicals.html`
- `AgriTech/farm_labor.html`
- `AgriTech/mandi.css`
- `AgriTech/rental.css`
- `AgriTech/mandi.js`
- `AgriTech/rental.js`

### Files to Modify
- `AgriTech/backend/api/v1/__init__.py` - Register new blueprints
- `AgriTech/backend/models/equipment.py` - Add new fields
- `AgriTech/backend/services/rental_service.py` - Enhance with new features
- `AgriTech/backend/models/__init__.py` - Import new models
- `AgriTech/app.py` - Register new routes
- `AgriTech/navbar.html` - Add navigation links

---

## Success Metrics

1. **Mandi Price Module**
   - API response time < 2 seconds
   - Cache hit rate > 80%
   - Price data freshness < 6 hours
   - All states and commodities covered

2. **Rental Module**
   - Booking conversion rate > 15%
   - Average equipment utilization > 40%
   - User satisfaction rating > 4.0/5.0
   - Response time for availability check < 500ms
