from django.shortcuts import render
from django.http import HttpResponse
from . import lstm as lstm_model

# Create your views here.
# view có thể trả về luôn 
def get_home(request):
    return render(request,'home.html')

def submit_form(request):
    if request.method == 'POST':
        content = request.POST.get('content')
        result = lstm_model.predict_email(content)
    return render(request, 'home.html', {'result': result})