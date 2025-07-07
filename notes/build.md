# Linux

## 가상 환경
    python3 -m venv .biostats
    source .biostats/bin/activate

## 저장소 복제
    git clone https://github.com/hikarimusic/BIOSTATS.git
    cd BIOSTATS
    pip install .
    biostats

## 패키지 빌드
    pip install --upgrade build
    python3 -m build

## 패키지 게시
    pip install --upgrade twine
    twine upload dist/*

## 실행 파일 빌드
    pip install --upgrade pyinstaller
    python3 pyinstaller.py


# Windows

## 가상 환경
    py -m venv .biostats
    .biostats\Scriptsctivate.bat

## 저장소 복제
    git clone https://github.com/hikarimusic/BIOSTATS.git
    cd BIOSTATS
    pip install .
    biostats

## 패키지 빌드
    pip install --upgrade build
    py -m build

## 실행 파일 빌드
    pip install --upgrade pyinstaller
    py pyinstaller.py
