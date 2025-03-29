from django.shortcuts import render

# Create your views here.
def get_home(request):
    return render(request,'home.html')

def submit_form(request):
    if request.method == 'POST':
        content = request.POST.get('content')
        print(f'Nội dung: {content}')
    return render(request, 'index.html')
