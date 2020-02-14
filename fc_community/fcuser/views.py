from django.http import HttpResponse
from django.shortcuts import render
from django.contrib.auth.hashers import make_password  # check_password 를 사용해 비밀번호 확인도 가능함

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
        # username = request.POST["username"]
        # password = request.POST["password"]
        # re_password = request.POST["re-password"]

        # # ex 4. 입력받은 값에 대한 예외처리 구현
        # username = request.POST.get("username", None)  # None 대신 거짓에 대한 값이어도 상관없음
        # password = request.POST.get("password", None)
        # re_password = request.POST.get("re-password", None)

        # Quiz. 이메일 필드 추가하기
        username = request.POST.get("username", None)
        useremail = request.POST.get("useremail", None)
        password = request.POST.get("password", None)
        re_password = request.POST.get("re-password", None)

        # ex 1. 에러 메세지 출력
        # if password != re_password:
        #     return HttpResponse("비밀번호가 다릅니다.")
        #
        # fcuser = Fcuser(username=username, password=password)
        # fcuser.save()
        #
        # return render(request, 'register.html')

        # ex 2. render 함수에 값 전달하기
        # res_data = {}
        # if password != re_password:
        #     res_data['error'] = '비밀번호가 다릅니다.'
        # else:
        #     fcuser = Fcuser(username=username, password=password)
        #     fcuser.save()
        #
        # return render(request, 'register.html', res_data)

        ## 작업 후 register.html 에서 {{ error }} 에 해당하는 부분과 연결됨

        # ex 3. 비밀번호 암호화하기
        # from django.contrib.auth.hashers import make_password 가 선언되었는지 반드시 확인!
        # 비밀번호 확인
        # res_data = {}
        # if password != re_password:
        #     res_data['error'] = "비밀번호가 다릅니다."
        # else:
        #     # 객체 생성 및 DB에 저장
        #     fcuser = Fcuser(username=username, password=make_password(password))
        #     fcuser.save()
        #
        # return render(request, 'register.html', res_data) # 만약 폴더 내에 존재한다면
                                                    # folder명/register.html 형태로 작성

        # ex 4. 입력받은 값에 대한 예외처리 구현
        # res_data = {}
        #
        # if not (username and password and re_password):
        #     res_data['error'] = "모든 값을 입력해야합니다."
        # elif password != re_password:
        #     res_data['error'] = "비밀번호가 다릅니다."
        # else:
        #     # 객체 생성 및 DB에 저장
        #     fcuser = Fcuser(username=username, password=make_password(password))
        #     fcuser.save()

        # Quiz . 이메일 필드 추가하기
        res_data = {}

        if not (username and useremail and password and re_password):
            res_data['error'] = "모든 값을 입력해야합니다."
        elif password != re_password:
            res_data['error'] = "비밀번호가 다릅니다."
        else:
            # 객체 생성 및 DB에 저장
            fcuser = Fcuser(username=username, password=make_password(password), useremail=useremail)
            fcuser.save()

        return render(request, 'register.html', res_data)