from django.db import models

# Create your models here.
class Fcuser(models.Model):
    # id 는 자동으로 생성됨
    username = models.CharField(max_length=32, verbose_name='사용자명')                       # 사용자명
    useremail = models.EmailField(max_length=128, verbose_name='사용자이메일')               # 사용자 이메일
    password = models.CharField(max_length=64, verbose_name='비밀번호')                      # 비밀번호
    registerd_dttm = models.DateTimeField(auto_now_add=True, verbose_name='등록시간')       # 등록시간
    # 등록시간 / auto_now_add 옵션 : 객체(클래스)가 저장되는 시점의 시간으로 기록됨

    # class 객체가 문자열로 변환되는 경우 어떤 방식으로 변환할 지를 결정하는 내장함수
    # 관리자 페이지에서 사용자의 이름을 한 번에 확인할 수 있음
    def __str__(self):
        return self.username

    # 사용자 지정 옵션
    class Meta:
        db_table = 'fc_user'  # 테이블명 지정
        verbose_name = "사용자" # 메인 페이지에 나오는 모델의 명칭을 변경하기 위한 방법
        verbose_name_plural = "사용자" # 메인 페이지에 나오는 모델의 명칭을 변경하기 위한 방법
                                        # django 에서는 모델명을 표시할 때 기본적으로 복수를
                                        # 사용하기 때문에 verbose_name_plural 옵션을 사용하는 것

    # 만약 값 혹은 필드가 수정(추가, 변경, 삭제 등)되는 경우
    # 수정될 때마다 makemigrations -> migrate 과정을 실행함으로써
    # 데이터베이스를 자동으로 관리할 수 있다.