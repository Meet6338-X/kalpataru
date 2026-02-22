"""
MandiSense Database Models
Market Intelligence Engine for Agricultural Prices

Models for:
- Price Alerts
- Price Forecasting
- Demand Analytics
- Best Mandi Recommendations
- FPO Trade Analytics
"""

from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, Text, JSON, Date, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class MandiPriceAlert(Base):
    """Price alert notifications for farmers and traders"""
    __tablename__ = 'mandi_price_alerts'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, nullable=False)
    commodity = Column(String(100), nullable=False)
    state = Column(String(100), nullable=True)
    district = Column(String(100), nullable=True)
    target_price = Column(Float, nullable=False)
    current_price = Column(Float, nullable=True)
    price_unit = Column(String(20), default='Quintal')
    is_active = Column(Boolean, default=True)
    notification_method = Column(String(20), default='APP')
    phone_number = Column(String(20), nullable=True)
    last_checked_at = Column(DateTime, nullable=True)
    triggered_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'commodity': self.commodity,
            'state': self.state,
            'district': self.district,
            'target_price': self.target_price,
            'current_price': self.current_price,
            'is_active': self.is_active,
            'notification_method': self.notification_method,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


class MandiPriceForecast(Base):
    """Price forecasts for commodities using ML models"""
    __tablename__ = 'mandi_price_forecasts'
    
    id = Column(Integer, primary_key=True)
    commodity = Column(String(100), nullable=False)
    state = Column(String(100), nullable=True)
    district = Column(String(100), nullable=True)
    market = Column(String(100), nullable=True)
    forecast_date = Column(Date, nullable=False)
    predicted_min_price = Column(Float, nullable=True)
    predicted_max_price = Column(Float, nullable=True)
    predicted_modal_price = Column(Float, nullable=True)
    confidence_score = Column(Float, nullable=True)
    forecast_horizon = Column(Integer, nullable=False)
    model_version = Column(String(20), default='v1.0')
    methodology = Column(String(50), default='LINEAR_REGRESSION')
    price_unit = Column(String(20), default='Quintal')
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'commodity': self.commodity,
            'forecast_date': self.forecast_date.isoformat() if self.forecast_date else None,
            'predicted_modal_price': self.predicted_modal_price,
            'confidence_score': self.confidence_score,
            'forecast_horizon': self.forecast_horizon
        }


class MandiDemandInsight(Base):
    """Demand analytics and insights for commodities"""
    __tablename__ = 'mandi_demand_insights'
    
    id = Column(Integer, primary_key=True)
    commodity = Column(String(100), nullable=False)
    state = Column(String(100), nullable=True)
    district = Column(String(100), nullable=True)
    demand_score = Column(Float, nullable=True)
    supply_score = Column(Float, nullable=True)
    trend = Column(String(20), nullable=True)
    trend_percentage = Column(Float, nullable=True)
    peak_season_start = Column(String(10), nullable=True)
    peak_season_end = Column(String(10), nullable=True)
    avg_arrival_volume = Column(Float, nullable=True)
    major_buyers = Column(JSON, nullable=True)
    price_volatility = Column(Float, nullable=True)
    analyzed_at = Column(DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'commodity': self.commodity,
            'region': f"{self.district}, {self.state}" if self.district else self.state,
            'demand_score': self.demand_score,
            'trend': self.trend,
            'analyzed_at': self.analyzed_at.isoformat() if self.analyzed_at else None
        }


class MandiBestRecommendation(Base):
    """Best mandi recommendations for farmers"""
    __tablename__ = 'mandi_recommendations'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, nullable=True)
    commodity = Column(String(100), nullable=False)
    quantity_tons = Column(Float, nullable=False)
    origin_state = Column(String(100), nullable=False)
    origin_district = Column(String(100), nullable=True)
    origin_market = Column(String(100), nullable=True)
    
    recommended_market_1 = Column(String(100), nullable=True)
    recommended_price_1 = Column(Float, nullable=True)
    recommended_distance_1 = Column(Float, nullable=True)
    estimated_transport_cost_1 = Column(Float, nullable=True)
    net_profit_1 = Column(Float, nullable=True)
    
    recommended_market_2 = Column(String(100), nullable=True)
    recommended_price_2 = Column(Float, nullable=True)
    recommended_distance_2 = Column(Float, nullable=True)
    estimated_transport_cost_2 = Column(Float, nullable=True)
    net_profit_2 = Column(Float, nullable=True)
    
    recommended_market_3 = Column(String(100), nullable=True)
    recommended_price_3 = Column(Float, nullable=True)
    recommended_distance_3 = Column(Float, nullable=True)
    estimated_transport_cost_3 = Column(Float, nullable=True)
    net_profit_3 = Column(Float, nullable=True)
    
    calculation_notes = Column(Text, nullable=True)
    calculated_at = Column(DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'commodity': self.commodity,
            'quantity_tons': self.quantity_tons,
            'origin_state': self.origin_state,
            'recommendations': [
                {'rank': 1, 'market': self.recommended_market_1, 'price': self.recommended_price_1, 
                 'distance_km': self.recommended_distance_1, 'transport_cost': self.estimated_transport_cost_1,
                 'net_profit': self.net_profit_1},
                {'rank': 2, 'market': self.recommended_market_2, 'price': self.recommended_price_2,
                 'distance_km': self.recommended_distance_2, 'transport_cost': self.estimated_transport_cost_2,
                 'net_profit': self.net_profit_2},
                {'rank': 3, 'market': self.recommended_market_3, 'price': self.recommended_price_3,
                 'distance_km': self.recommended_distance_3, 'transport_cost': self.estimated_transport_cost_3,
                 'net_profit': self.net_profit_3}
            ]
        }


class FPOTradeAnalytics(Base):
    """Trade analytics for Farmer Producer Organizations"""
    __tablename__ = 'fpo_trade_analytics'
    
    id = Column(Integer, primary_key=True)
    fpo_id = Column(Integer, nullable=False)
    commodity = Column(String(100), nullable=False)
    total_volume_sold = Column(Float, default=0)
    total_transactions = Column(Integer, default=0)
    avg_selling_price = Column(Float, default=0)
    highest_price = Column(Float, default=0)
    lowest_price = Column(Float, default=0)
    price_vs_mandi_avg = Column(Float, default=0)
    preferred_markets = Column(JSON, nullable=True)
    preferred_buyers = Column(JSON, nullable=True)
    seasonal_performance = Column(JSON, nullable=True)
    period_start = Column(Date, nullable=False)
    period_end = Column(Date, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'fpo_id': self.fpo_id,
            'commodity': self.commodity,
            'total_volume_sold': self.total_volume_sold,
            'avg_selling_price': self.avg_selling_price,
            'period_start': self.period_start.isoformat() if self.period_start else None,
            'period_end': self.period_end.isoformat() if self.period_end else None
        }


class FarmLinkListing(Base):
    """Farmer produce listings for direct sale to buyers"""
    __tablename__ = 'farmlink_listings'
    
    id = Column(Integer, primary_key=True)
    listing_id = Column(String(50), unique=True, nullable=False)
    farmer_id = Column(Integer, nullable=False)
    crop_name = Column(String(100), nullable=False)
    quantity_tons = Column(Float, nullable=False)
    price_per_ton = Column(Float, nullable=False)
    location = Column(String(200), nullable=False)
    state = Column(String(100), nullable=False)
    district = Column(String(100), nullable=True)
    description = Column(Text, nullable=True)
    image_url = Column(String(500), nullable=True)
    status = Column(String(20), default='ACTIVE')
    
    # Quality fields
    quality_grade = Column(String(10), nullable=True)
    quality_photos = Column(JSON, nullable=True)
    third_party_verified = Column(Boolean, default=False)
    verification_agency = Column(String(100), nullable=True)
    quality_score = Column(Float, nullable=True)
    
    # Harvest & Storage
    harvest_date = Column(Date, nullable=True)
    expiry_date = Column(Date, nullable=True)
    storage_type = Column(String(50), nullable=True)
    
    # FPO
    fpo_id = Column(Integer, nullable=True)
    is_aggregated = Column(Boolean, default=False)
    member_count = Column(Integer, default=1)
    
    eligible_for_gov_procurement = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'listing_id': self.listing_id,
            'farmer_id': self.farmer_id,
            'crop_name': self.crop_name,
            'quantity_tons': self.quantity_tons,
            'price_per_ton': self.price_per_ton,
            'location': self.location,
            'state': self.state,
            'status': self.status,
            'quality': {
                'grade': self.quality_grade,
                'score': self.quality_score,
                'verified': self.third_party_verified
            },
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


class FarmLinkOffer(Base):
    """Buyer offers on farmer listings"""
    __tablename__ = 'farmlink_offers'
    
    id = Column(Integer, primary_key=True)
    offer_id = Column(String(50), unique=True, nullable=False)
    listing_id = Column(Integer, nullable=False)
    buyer_id = Column(Integer, nullable=False)
    offered_price = Column(Float, nullable=False)
    message = Column(Text, nullable=True)
    status = Column(String(20), default='PENDING')
    buyer_type = Column(String(50), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'offer_id': self.offer_id,
            'listing_id': self.listing_id,
            'buyer_id': self.buyer_id,
            'offered_price': self.offered_price,
            'status': self.status,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


class FarmLinkDeal(Base):
    """Completed deals between farmers and buyers"""
    __tablename__ = 'farmlink_deals'
    
    id = Column(Integer, primary_key=True)
    deal_id = Column(String(50), unique=True, nullable=False)
    listing_id = Column(Integer, nullable=False)
    offer_id = Column(Integer, nullable=False)
    farmer_id = Column(Integer, nullable=False)
    buyer_id = Column(Integer, nullable=False)
    final_price = Column(Float, nullable=False)
    quantity_tons = Column(Float, nullable=False)
    total_value = Column(Float, nullable=False)
    status = Column(String(20), default='PENDING')
    delivery_address = Column(String(500), nullable=True)
    payment_status = Column(String(20), default='PENDING')
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    
    def to_dict(self):
        return {
            'id': self.id,
            'deal_id': self.deal_id,
            'farmer_id': self.farmer_id,
            'buyer_id': self.buyer_id,
            'final_price': self.final_price,
            'quantity_tons': self.quantity_tons,
            'total_value': self.total_value,
            'status': self.status,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


class FarmLinkForwardContract(Base):
    """Forward contracts for price locking"""
    __tablename__ = 'farmlink_forward_contracts'
    
    id = Column(Integer, primary_key=True)
    contract_id = Column(String(50), unique=True, nullable=False)
    farmer_id = Column(Integer, nullable=False)
    buyer_id = Column(Integer, nullable=False)
    commodity = Column(String(100), nullable=False)
    quantity_tons = Column(Float, nullable=False)
    locked_price_per_ton = Column(Float, nullable=False)
    delivery_start_date = Column(Date, nullable=False)
    delivery_end_date = Column(Date, nullable=False)
    status = Column(String(20), default='PENDING')
    total_delivered = Column(Float, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'contract_id': self.contract_id,
            'commodity': self.commodity,
            'quantity_tons': self.quantity_tons,
            'locked_price': self.locked_price_per_ton,
            'status': self.status,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


class FarmLinkEscrow(Base):
    """Escrow payment system for secure transactions"""
    __tablename__ = 'farmlink_escrow'
    
    id = Column(Integer, primary_key=True)
    escrow_id = Column(String(50), unique=True, nullable=False)
    deal_id = Column(Integer, nullable=True)
    amount = Column(Float, nullable=False)
    status = Column(String(20), default='HELD')
    held_at = Column(DateTime, default=datetime.utcnow)
    released_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'escrow_id': self.escrow_id,
            'deal_id': self.deal_id,
            'amount': self.amount,
            'status': self.status,
            'held_at': self.held_at.isoformat() if self.held_at else None
        }


class FPOMember(Base):
    """FPO (Farmer Producer Organization) membership"""
    __tablename__ = 'fpo_members'
    
    id = Column(Integer, primary_key=True)
    fpo_id = Column(Integer, nullable=False)
    farmer_id = Column(Integer, nullable=False)
    member_since = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    share_count = Column(Integer, default=1)
    role = Column(String(20), default='MEMBER')
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'fpo_id': self.fpo_id,
            'farmer_id': self.farmer_id,
            'is_active': self.is_active,
            'role': self.role,
            'member_since': self.member_since.isoformat() if self.member_since else None
        }


class GovProcurementBid(Base):
    """Government procurement bids (FCI, NAFED, PM-AASHA)"""
    __tablename__ = 'gov_procurement_bids'
    
    id = Column(Integer, primary_key=True)
    bid_id = Column(String(50), unique=True, nullable=False)
    listing_id = Column(Integer, nullable=False)
    agency = Column(String(100), nullable=False)
    quantity_quintals = Column(Float, nullable=False)
    price_per_quintal = Column(Float, nullable=False)
    total_bid_value = Column(Float, nullable=True)
    status = Column(String(20), default='SUBMITTED')
    payment_status = Column(String(20), default='PENDING')
    submitted_at = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'bid_id': self.bid_id,
            'listing_id': self.listing_id,
            'agency': self.agency,
            'quantity': self.quantity_quintals,
            'price_per_quintal': self.price_per_quintal,
            'status': self.status,
            'submitted_at': self.submitted_at.isoformat() if self.submitted_at else None
        }
