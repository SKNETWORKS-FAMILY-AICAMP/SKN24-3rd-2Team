import requests

BASE_URL = "https://api.jolpi.ca/ergast/f1"

def get_seasons():
    """역대 시즌 목록 반환"""
    response = requests.get(f"{BASE_URL}/seasons")
    return response.json()


def get_circuits():
    """서킷 목록 반환"""
    response = requests.get(f"{BASE_URL}/circuits")
    return response.json()


def get_races(year):
    """시즌 레이스 일정, 결과 반환"""
    response = requests.get(f"{BASE_URL}/{year}/races")
    return response.json()


def get_constructors(year):
    """컨스트럭터 목록 반환"""
    response = requests.get(f"{BASE_URL}/{year}/constructors")
    return response.json()


def get_drivers(year):
    """드라이버 목록 반환"""
    response = requests.get(f"{BASE_URL}/{year}/drivers")
    return response.json()


def get_results(year, round=None):
    """레이스 결과 반환"""
    if round:
        response = requests.get(f"{BASE_URL}/{year}/{round}/results")
    else:
        response = requests.get(f"{BASE_URL}/{year}/results")
    return response.json()


def get_sprint(year, round=None):
    """스프린트 결과 반환"""
    if round:
        response = requests.get(f"{BASE_URL}/{year}/{round}/sprint")
    else: # 시즌 전체 스프린트 결과 
        response = requests.get(f"{BASE_URL}/{year}/sprint")
    return response.json()


def get_qualifying(year, round=None):
    """퀄리파잉 결과 반환"""
    if round:
        response = requests.get(f"{BASE_URL}/{year}/{round}/qualifying")
    else: # 시즌 전체 퀄리파잉 결과
        response = requests.get(f"{BASE_URL}/{year}/qualifying")
    return response.json()


def get_pitstops(year, round):
    """피트 스탑 기록 반환"""
    response = requests.get(f"{BASE_URL}/{year}/{round}/pitstops")
    return response.json()


def get_laps(year, round):
    """랩타임 반환"""
    response = requests.get(f"{BASE_URL}/{year}/{round}/laps")
    return response.json()


def get_driver_standings(year):
    """드라이버 챔피언십 순위 반환"""
    response = requests.get(f"{BASE_URL}/{year}/driverstandings")
    return response.json()


def get_constructor_standings(year):
    """컨스트럭터 챔피언십 순위 반환"""
    response = requests.get(f"{BASE_URL}/{year}/constructorstandings")
    return response.json()


def get_status():
    """리타이어 사유 반환"""
    response = requests.get(f"{BASE_URL}/status")
    return response.json()


def get_season_data(year):
    """시즌별 데이터 수집"""
    races = get_races(year)
    results = get_results(year)
    driver_standings = get_driver_standings(year)
    constructor_standings = get_constructor_standings(year)
    qualifying = get_qualifying(year)

    return {
        "year": year,
        "races": races,
        "results": results,
        "qualifying": qualifying,
        "driver_standings": driver_standings,
        "constructor_standings": constructor_standings,
    }


def get_round_data(year, round):
    """특정 라운드 데이터"""
    return {
        "results": get_results(year, round),
        "qualifying": get_qualifying(year, round),
        "pitstops": get_pitstops(year, round),
        "sprint": get_sprint(year, round),
        "laps": get_laps(year, round),
    }