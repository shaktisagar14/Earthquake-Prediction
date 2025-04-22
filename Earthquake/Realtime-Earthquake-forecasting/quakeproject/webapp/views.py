from django.shortcuts import render
from datetime import datetime, timedelta
from .utils import prepare_earthquake_data_and_model, get_earth_quake_estimates

days_out_to_predict = 7
earthquake_live = prepare_earthquake_data_and_model(days_out_to_predict)

def index(request):
    if request.method == 'POST':
        horizon_int = int(request.POST.get('slider_date_horizon'))
        horizon_date = datetime.today() + timedelta(days=horizon_int)

        earthquake_data = get_earth_quake_estimates(str(horizon_date)[:10], earthquake_live)

        return render(request, 'index.html', {
            'date_horizon': horizon_date.strftime('%m/%d/%Y'),
            'earthquake_horizon': earthquake_data,
            'current_value': horizon_int,
            'days_out_to_predict': days_out_to_predict,
        })

    return render(request, 'index.html', {
        'date_horizon': datetime.today().strftime('%m/%d/%Y'),
        'earthquake_horizon': '',
        'current_value': 0,
        'days_out_to_predict': days_out_to_predict,
    })
