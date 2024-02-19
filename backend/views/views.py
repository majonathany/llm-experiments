from django.shortcuts import render        
from django.http import HttpResponse

def initialize(request):    
    return HttpResponse('<h1>This is about me!.</h1>')  

def test_connection_to_db(request):
    from ..llm import rag
    return HttpResponse(f'<h1>This is about {rag.test_connection_to_db("hi")}!.</h1>')  
