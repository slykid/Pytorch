from django.contrib import admin
from polls.models import Question, Choice

# 19.04.04 수정
# class ChoiceInline(admin.StackedInline):
class ChoiceInline(admin.TabularInline):
    model = Choice
    extra = 2

# 19.04.04 수정
class QuestionAdmin(admin.ModelAdmin):
    # fields = ["pub_date", "question_text"]
    fieldsets = [
        ('Question Statement', {'fields':['question_text']}),
        # ('Date Infomation', {'fields':['pub_date']})
        ('Date Infomation', {'fields': ['pub_date'], 'classes': ['collapse']})
    ]
    inlines = [ChoiceInline]
    list_display = ("question_text", "pub_date")
    list_filter = ['pub_date']
    search_fields = ['question_text']

# Register your models here.
#admin.site.register(Question)
admin.site.register(Question, QuestionAdmin) # 19.04.04 수정
admin.site.register(Choice)
