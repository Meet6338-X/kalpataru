"""
Nutrient Service for Kalpataru.
Scientific formulas for soil health analysis and fertilizer recommendations.
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

from utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class NutrientLevel:
    """Data class for nutrient level information."""
    name: str
    symbol: str
    current_value: float
    unit: str
    status: str  # 'low', 'optimal', 'high'
    target_value: float


class NutrientService:
    """
    Service for soil nutrient analysis and fertilizer recommendations.
    Provides scientific calculations for nutrient management.
    """
    
    # NPK targets for different crops (kg/ha or mg/kg)
    CROP_NPK_TARGETS = {
        'wheat': {'N': 150, 'P': 60, 'K': 120},
        'rice': {'N': 120, 'P': 50, 'K': 100},
        'maize': {'N': 180, 'P': 75, 'K': 150},
        'cotton': {'N': 100, 'P': 50, 'K': 80},
        'sugarcane': {'N': 250, 'P': 125, 'K': 200},
        'potato': {'N': 150, 'P': 100, 'K': 200},
        'tomato': {'N': 100, 'P': 80, 'K': 120},
        'onion': {'N': 100, 'P': 60, 'K': 80},
        'soybean': {'N': 30, 'P': 60, 'K': 80},
        'groundnut': {'N': 25, 'P': 50, 'K': 60},
        'mustard': {'N': 80, 'P': 40, 'K': 40},
        'barley': {'N': 80, 'P': 40, 'K': 60},
        'chickpea': {'N': 20, 'P': 40, 'K': 40},
        'vegetables': {'N': 120, 'P': 80, 'K': 100},
        'fruits': {'N': 150, 'P': 100, 'K': 150}
    }
    
    # Secondary nutrient targets
    SECONDARY_NUTRIENT_TARGETS = {
        'calcium': {'target': 1500, 'unit': 'mg/kg'},
        'magnesium': {'target': 300, 'unit': 'mg/kg'},
        'sulfur': {'target': 15, 'unit': 'mg/kg'}
    }
    
    # Micronutrient targets
    MICRONUTRIENT_TARGETS = {
        'iron': {'target': 5, 'unit': 'mg/kg'},
        'zinc': {'target': 1.5, 'unit': 'mg/kg'},
        'copper': {'target': 1.0, 'unit': 'mg/kg'},
        'manganese': {'target': 3.0, 'unit': 'mg/kg'},
        'boron': {'target': 0.5, 'unit': 'mg/kg'}
    }
    
    # Fertilizer nutrient content (%)
    FERTILIZER_CONTENT = {
        'Urea': {'N': 46, 'P': 0, 'K': 0},
        'DAP': {'N': 18, 'P': 46, 'K': 0},
        'MOP': {'N': 0, 'P': 0, 'K': 60},
        'SSP': {'N': 0, 'P': 16, 'K': 0},
        'TSP': {'N': 0, 'P': 46, 'K': 0},
        'NPK-10-26-26': {'N': 10, 'P': 26, 'K': 26},
        'NPK-20-20-20': {'N': 20, 'P': 20, 'K': 20},
        'NPK-12-32-16': {'N': 12, 'P': 32, 'K': 16},
        'Ammonium_Sulphate': {'N': 20.6, 'P': 0, 'K': 0},
        'Calcium_Ammonium_Nitrate': {'N': 25, 'P': 0, 'K': 0}
    }
    
    # Nutrient status thresholds
    NPK_THRESHOLDS = {
        'N': {'low': 250, 'optimal': 500, 'high': 800},
        'P': {'low': 15, 'optimal': 30, 'high': 50},
        'K': {'low': 150, 'optimal': 300, 'high': 500}
    }
    
    @staticmethod
    def calculate_nutrient_gap(current: float, target: float) -> float:
        """
        Calculate the gap between current soil nutrients and target levels.
        
        Args:
            current: Current nutrient level
            target: Target nutrient level for crop
            
        Returns:
            Nutrient gap (positive = deficiency, negative = excess)
        """
        gap = target - current
        return round(gap, 2)
    
    @staticmethod
    def calculate_fertilizer_amount(
        nutrient_need_kg_ha: float, 
        nutrient_percentage: float
    ) -> float:
        """
        Calculate the amount of fertilizer needed based on nutrient content.
        
        Args:
            nutrient_need_kg_ha: Nutrient requirement in kg/ha
            nutrient_percentage: Percentage of nutrient in fertilizer
            
        Returns:
            Amount of fertilizer needed in kg/ha
        """
        if nutrient_percentage <= 0:
            return 0
        amount = (nutrient_need_kg_ha / (nutrient_percentage / 100))
        return round(amount, 2)
    
    @staticmethod
    def calculate_lime_requirement(current_ph: float, target_ph: float = 6.5) -> float:
        """
        Estimate lime required (tons/hectare) to raise soil pH.
        Simplified formula for medium textured soils.
        
        Args:
            current_ph: Current soil pH
            target_ph: Target soil pH (default 6.5)
            
        Returns:
            Lime requirement in tons/ha
        """
        if current_ph >= target_ph:
            return 0.0
        
        gap = target_ph - current_ph
        # Buffer capacity factor for medium textured soil
        # Approximately 1.5 tons per 0.5 pH increase
        lime_needed = (gap / 0.5) * 1.5
        return round(lime_needed, 2)
    
    @staticmethod
    def calculate_gypsum_requirement(
        current_ph: float, 
        target_ph: float = 7.5,
        soil_type: str = 'clay'
    ) -> float:
        """
        Estimate gypsum requirement to lower soil pH (for sodic soils).
        
        Args:
            current_ph: Current soil pH
            target_ph: Target soil pH
            soil_type: Type of soil
            
        Returns:
            Gypsum requirement in tons/ha
        """
        if current_ph <= target_ph:
            return 0.0
        
        # Simplified calculation
        gap = current_ph - target_ph
        base_rate = 2.0 if soil_type == 'clay' else 1.5
        gypsum_needed = gap * base_rate
        return round(gypsum_needed, 2)
    
    @classmethod
    def get_crop_targets(cls, crop_type: str) -> Dict[str, float]:
        """
        Get standard N-P-K targets for a specific crop.
        
        Args:
            crop_type: Type of crop
            
        Returns:
            Dictionary with N, P, K targets
        """
        return cls.CROP_NPK_TARGETS.get(
            crop_type.lower(), 
            {'N': 100, 'P': 50, 'K': 80}
        )
    
    @classmethod
    def analyze_nutrient_status(
        cls, 
        nutrient: str, 
        value: float
    ) -> str:
        """
        Determine nutrient status based on value.
        
        Args:
            nutrient: Nutrient name (N, P, K)
            value: Current nutrient value
            
        Returns:
            Status string ('low', 'optimal', 'high')
        """
        thresholds = cls.NPK_THRESHOLDS.get(nutrient.upper())
        
        if not thresholds:
            return 'unknown'
        
        if value < thresholds['low']:
            return 'low'
        elif value <= thresholds['optimal']:
            return 'low-moderate'
        elif value <= thresholds['high']:
            return 'optimal'
        else:
            return 'high'
    
    @classmethod
    def calculate_npk_recommendation(
        cls,
        current_n: float,
        current_p: float,
        current_k: float,
        crop_type: str,
        area_hectares: float = 1.0
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive NPK fertilizer recommendation.
        
        Args:
            current_n: Current nitrogen level (kg/ha)
            current_p: Current phosphorus level (kg/ha)
            current_k: Current potassium level (kg/ha)
            crop_type: Target crop type
            area_hectares: Area in hectares
            
        Returns:
            Comprehensive fertilizer recommendation
        """
        # Get crop targets
        targets = cls.get_crop_targets(crop_type)
        
        # Calculate gaps
        n_gap = cls.calculate_nutrient_gap(current_n, targets['N'])
        p_gap = cls.calculate_nutrient_gap(current_p, targets['P'])
        k_gap = cls.calculate_nutrient_gap(current_k, targets['K'])
        
        # Determine status
        n_status = cls.analyze_nutrient_status('N', current_n)
        p_status = cls.analyze_nutrient_status('P', current_p)
        k_status = cls.analyze_nutrient_status('K', current_k)
        
        # Calculate fertilizer amounts
        recommendations = []
        
        if n_gap > 0:
            urea_amount = cls.calculate_fertilizer_amount(n_gap, 46)
            recommendations.append({
                'fertilizer': 'Urea',
                'amount_kg_per_ha': urea_amount,
                'total_amount_kg': round(urea_amount * area_hectares, 2),
                'nutrient_supplied': 'Nitrogen',
                'application_timing': 'Split into 2-3 doses during growth'
            })
        
        if p_gap > 0:
            dap_amount = cls.calculate_fertilizer_amount(p_gap, 46)
            recommendations.append({
                'fertilizer': 'DAP',
                'amount_kg_per_ha': dap_amount,
                'total_amount_kg': round(dap_amount * area_hectares, 2),
                'nutrient_supplied': 'Phosphorus',
                'application_timing': 'Apply at sowing/transplanting'
            })
        
        if k_gap > 0:
            mop_amount = cls.calculate_fertilizer_amount(k_gap, 60)
            recommendations.append({
                'fertilizer': 'MOP',
                'amount_kg_per_ha': mop_amount,
                'total_amount_kg': round(mop_amount * area_hectares, 2),
                'nutrient_supplied': 'Potassium',
                'application_timing': 'Apply during active growth phase'
            })
        
        return {
            'crop': crop_type,
            'area_hectares': area_hectares,
            'current_levels': {
                'N': {'value': current_n, 'status': n_status},
                'P': {'value': current_p, 'status': p_status},
                'K': {'value': current_k, 'status': k_status}
            },
            'target_levels': targets,
            'nutrient_gaps': {
                'N': n_gap,
                'P': p_gap,
                'K': k_gap
            },
            'recommendations': recommendations,
            'summary': cls._generate_summary(n_gap, p_gap, k_gap)
        }
    
    @classmethod
    def calculate_secondary_nutrients(
        cls,
        calcium: float = None,
        magnesium: float = None,
        sulfur: float = None
    ) -> Dict[str, Any]:
        """
        Analyze secondary nutrient levels.
        
        Args:
            calcium: Calcium level (mg/kg)
            magnesium: Magnesium level (mg/kg)
            sulfur: Sulfur level (mg/kg)
            
        Returns:
            Secondary nutrient analysis
        """
        results = {}
        
        if calcium is not None:
            target = cls.SECONDARY_NUTRIENT_TARGETS['calcium']['target']
            results['calcium'] = {
                'value': calcium,
                'target': target,
                'status': 'optimal' if calcium >= target else 'low',
                'gap': max(0, target - calcium)
            }
        
        if magnesium is not None:
            target = cls.SECONDARY_NUTRIENT_TARGETS['magnesium']['target']
            results['magnesium'] = {
                'value': magnesium,
                'target': target,
                'status': 'optimal' if magnesium >= target else 'low',
                'gap': max(0, target - magnesium)
            }
        
        if sulfur is not None:
            target = cls.SECONDARY_NUTRIENT_TARGETS['sulfur']['target']
            results['sulfur'] = {
                'value': sulfur,
                'target': target,
                'status': 'optimal' if sulfur >= target else 'low',
                'gap': max(0, target - sulfur)
            }
        
        return results
    
    @classmethod
    def calculate_micronutrients(
        cls,
        iron: float = None,
        zinc: float = None,
        copper: float = None,
        manganese: float = None,
        boron: float = None
    ) -> Dict[str, Any]:
        """
        Analyze micronutrient levels.
        
        Args:
            iron: Iron level (mg/kg)
            zinc: Zinc level (mg/kg)
            copper: Copper level (mg/kg)
            manganese: Manganese level (mg/kg)
            boron: Boron level (mg/kg)
            
        Returns:
            Micronutrient analysis
        """
        results = {}
        inputs = {
            'iron': iron,
            'zinc': zinc,
            'copper': copper,
            'manganese': manganese,
            'boron': boron
        }
        
        for nutrient, value in inputs.items():
            if value is not None:
                target = cls.MICRONUTRIENT_TARGETS[nutrient]['target']
                results[nutrient] = {
                    'value': value,
                    'target': target,
                    'status': 'optimal' if value >= target else 'deficient',
                    'gap': max(0, target - value)
                }
        
        return results
    
    @staticmethod
    def _generate_summary(n_gap: float, p_gap: float, k_gap: float) -> str:
        """Generate a summary of nutrient recommendations."""
        gaps = {'Nitrogen': n_gap, 'Phosphorus': p_gap, 'Potassium': k_gap}
        
        deficiencies = [n for n, g in gaps.items() if g > 0]
        
        if not deficiencies:
            return "Soil nutrient levels are adequate for the target crop."
        
        if len(deficiencies) == 1:
            return f"{deficiencies[0]} supplementation required."
        elif len(deficiencies) == 2:
            return f"{deficiencies[0]} and {deficiencies[1]} supplementation required."
        else:
            return "Complete NPK fertilization program recommended."


def analyze_nutrients(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main function to analyze soil nutrients.
    
    Args:
        data: Dictionary containing:
            - N: Nitrogen level
            - P: Phosphorus level
            - K: Potassium level
            - crop_type: Target crop
            - area_hectares: Area (optional)
            - pH: Soil pH (optional)
            - secondary: Dict of secondary nutrients (optional)
            - micro: Dict of micronutrients (optional)
            
    Returns:
        Comprehensive nutrient analysis
    """
    service = NutrientService()
    
    # Main NPK analysis
    result = service.calculate_npk_recommendation(
        current_n=data.get('N', data.get('nitrogen', 50)),
        current_p=data.get('P', data.get('phosphorus', 30)),
        current_k=data.get('K', data.get('potassium', 40)),
        crop_type=data.get('crop_type', 'wheat'),
        area_hectares=data.get('area_hectares', 1.0)
    )
    
    # pH analysis
    if 'pH' in data or 'ph' in data:
        ph = data.get('pH', data.get('ph', 7.0))
        if ph < 6.5:
            result['ph_adjustment'] = {
                'current_ph': ph,
                'target_ph': 6.5,
                'lime_required_tons_ha': service.calculate_lime_requirement(ph)
            }
        elif ph > 7.5:
            result['ph_adjustment'] = {
                'current_ph': ph,
                'target_ph': 7.5,
                'gypsum_required_tons_ha': service.calculate_gypsum_requirement(ph)
            }
        else:
            result['ph_adjustment'] = {
                'current_ph': ph,
                'status': 'optimal'
            }
    
    # Secondary nutrients
    if 'secondary' in data:
        result['secondary_nutrients'] = service.calculate_secondary_nutrients(
            **data['secondary']
        )
    
    # Micronutrients
    if 'micro' in data:
        result['micronutrients'] = service.calculate_micronutrients(
            **data['micro']
        )
    
    return result


def get_fertilizer_schedule(
    crop_type: str,
    n_gap: float,
    p_gap: float,
    k_gap: float
) -> List[Dict[str, Any]]:
    """
    Get detailed fertilizer application schedule.
    
    Args:
        crop_type: Type of crop
        n_gap: Nitrogen gap
        p_gap: Phosphorus gap
        k_gap: Potassium gap
        
    Returns:
        List of scheduled applications
    """
    schedule = []
    
    # Base schedule patterns
    if crop_type.lower() in ['rice', 'wheat', 'maize']:
        # Cereals: Split N applications
        if n_gap > 0:
            schedule.extend([
                {
                    'stage': 'Basal',
                    'fertilizer': 'DAP + Urea',
                    'nitrogen_pct': 25,
                    'timing': 'At sowing'
                },
                {
                    'stage': 'Tillering',
                    'fertilizer': 'Urea',
                    'nitrogen_pct': 35,
                    'timing': '21 days after sowing'
                },
                {
                    'stage': 'Flowering',
                    'fertilizer': 'Urea',
                    'nitrogen_pct': 40,
                    'timing': '45 days after sowing'
                }
            ])
    else:
        # General schedule
        if n_gap > 0:
            schedule.extend([
                {
                    'stage': 'Basal',
                    'fertilizer': 'Urea',
                    'nitrogen_pct': 50,
                    'timing': 'At planting'
                },
                {
                    'stage': 'Top Dressing',
                    'fertilizer': 'Urea',
                    'nitrogen_pct': 50,
                    'timing': '30 days after planting'
                }
            ])
    
    # P and K typically applied at planting
    if p_gap > 0:
        schedule.append({
            'stage': 'Basal',
            'fertilizer': 'DAP or SSP',
            'phosphorus_pct': 100,
            'timing': 'At planting'
        })
    
    if k_gap > 0:
        schedule.append({
            'stage': 'Basal',
            'fertilizer': 'MOP',
            'potassium_pct': 100,
            'timing': 'At planting'
        })
    
    return schedule
