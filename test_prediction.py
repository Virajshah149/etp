import sys
sys.path.insert(0, 'src')
from prediction import HybridPredictor

p = HybridPredictor()
p.load_models()

scenarios = [
    dict(hour=8,  day_of_week=1, month=3, is_holiday=0, event_weight=0.0,
         temperature_c=25.0, weather_condition='Sunny',      rainfall_mm=0.0,  avg_demand_estimate=80.0),
    dict(hour=18, day_of_week=4, month=6, is_holiday=0, event_weight=0.0,
         temperature_c=24.0, weather_condition='Heavy Rain', rainfall_mm=15.0, avg_demand_estimate=100.0),
    dict(hour=12, day_of_week=6, month=1, is_holiday=1, event_weight=0.8,
         temperature_c=22.0, weather_condition='Sunny',      rainfall_mm=0.0,  avg_demand_estimate=120.0),
    dict(hour=2,  day_of_week=3, month=5, is_holiday=0, event_weight=0.0,
         temperature_c=27.0, weather_condition='Overcast',   rainfall_mm=2.0,  avg_demand_estimate=50.0),
]
labels = ['Weekday Morning Rush', 'Friday Rainy Evening', 'Sunday Holiday Event', 'Midweek Off-Peak Night']

print('--- Hybrid Prediction Smoke Test ---')
all_ok = True
for label, s in zip(labels, scenarios):
    r = p.predict(**s)
    lstm = r['lstm_prediction']
    rf   = r['rf_prediction']
    hyb  = r['hybrid_prediction']
    cat  = r['demand_category']
    print(label + ' => LSTM=' + str(lstm) + '  RF=' + str(rf) + '  Hybrid=' + str(hyb) + '  [' + cat + ']')
    if hyb <= 0:
        all_ok = False
        print('  ERROR: hybrid prediction is zero or negative!')

print()
print('SMOKE TEST PASSED' if all_ok else 'SMOKE TEST FAILED')
