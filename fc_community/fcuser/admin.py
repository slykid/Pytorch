from django.contrib import admin
from .models import Fcuser
# Register your models here.

class FcuserAdmin(admin.ModelAdmin):
    # 명시하고자 하는 정보를 표시하기 위한 방법
    list_display = ('username', 'password')

admin.site.register(Fcuser, FcuserAdmin)



