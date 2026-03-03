import pytest
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as c:
        yield c

def test_health(client):
    r = client.get('/api/health')
    assert r.status_code == 200
    d = json.loads(r.data)
    assert d['status'] == 'ok'

def test_analyze_single(client):
    r = client.post('/api/analyze',
                    json={'text': 'This is amazing!',
                          'username': 'test_user'},
                    content_type='application/json')
    assert r.status_code == 200
    d = json.loads(r.data)
    assert 'sentiment' in d
    assert d['sentiment'] in ['positive','negative','neutral']
    assert 'score' in d
    assert 'is_toxic' in d

def test_analyze_negative(client):
    r = client.post('/api/analyze',
                    json={'text': 'You are such an idiot!',
                          'username': 'test_user'},
                    content_type='application/json')
    assert r.status_code == 200
    d = json.loads(r.data)
    assert d['sentiment'] == 'negative'
    assert d['is_toxic'] == True

def test_analyze_empty(client):
    r = client.post('/api/analyze',
                    json={'text': ''},
                    content_type='application/json')
    assert r.status_code == 400

def test_batch_analyze(client):
    comments = [
        {'text': 'Amazing product!',  'username': 'user1'},
        {'text': 'Terrible quality!', 'username': 'user2'},
        {'text': 'Okay I guess.',     'username': 'user3'},
    ]
    r = client.post('/api/analyze/batch',
                    json={'comments': comments},
                    content_type='application/json')
    assert r.status_code == 200
    d = json.loads(r.data)
    assert d['total'] == 3
    assert 'summary' in d
    assert 'results' in d

def test_stats(client):
    r = client.get('/api/stats')
    assert r.status_code == 200
    d = json.loads(r.data)
    assert 'total' in d
    assert 'positive' in d
    assert 'negative' in d
    assert 'toxic' in d

def test_history(client):
    r = client.get('/api/history?limit=10')
    assert r.status_code == 200
    assert isinstance(json.loads(r.data), list)

def test_toxic(client):
    r = client.get('/api/toxic')
    assert r.status_code == 200
    assert isinstance(json.loads(r.data), list)

def test_preprocess(client):
    r = client.post('/api/preprocess',
                    json={'text': 'This is AMAZING!!! ❤️'},
                    content_type='application/json')
    assert r.status_code == 200
    d = json.loads(r.data)
    assert 'cleaned' in d
    assert 'is_toxic' in d
    assert 'is_sarcastic' in d