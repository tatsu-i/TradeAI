# 機械学習を使ったトレードプラットホーム

## ビルド
```
$ make
```

## インストール
```
copy dist\windows\libstock.dll C:\Program Files (x86)\TradeStation 9.5\Program
```

# dockerのビルドと起動
```
$ echo never |sudo tee -a /sys/kernel/mm/transparent_hugepage/enabled
$ docker-compose build
$ docker-compose up -d
```
