from django.http import HttpResponse
from django.shortcuts import render
from django.contrib.auth.hashers import make_password
from .models import Fcuser

# Create your views here.
def register(request):
    # register에 접근 방식이 2가지임
    # 1. GET
    if request.method == "GET":
        return render(request, 'register.html')
    # 2. POST
    elif request.method == "POST":
        # 회원가입을 위한 코드
        username = request.POST["username"]
        password = request.POST["password"]
        re_password = request.POST["re-password"]

        # 비밀번호 확인
        res_data = {}
        if password != re_password:
            res_data['error'] = "비밀번호가 다릅니다."
        else:
            # 객체 생성 및 DB에 저장
            fcuser = Fcuser(username=username, password=make_password(password))
            fcuser.save()

        return render(request, 'register.html', res_data) # 만약 폴더 내에 존재한다면
                                                    # folder명/register.html 형태로 작성

