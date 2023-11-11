# ReId
Скрипт для снятия метрики. Испольуем recall. Смотрим сколько было опознано правильно среди опознанного.
Для взятия изображений, чтобы снять идеальную модель берем папку query из линка https://drive.google.com/drive/folders/1e3tsTW-KPMISlWRrT18GqezbSuinbVh_?usp=sharing
идентично обрезаем поисковый предмет. Посчитать можно только для одного объекта. ЧТобы использоватьб для нескольких камер, просто передаем изображение из нескольких камер
Вручную указываем ожидаемый класс
тип имени {номер_класса}_{номер_фото_этого_класса}.jpg
по умолчанию используем по 4 изображения для каждого класса
для верности искомый класс помещаем в середину
по этой ссылке можно взять веса разных моделей в архиве model и логи по обучению, внутри каждой папки https://drive.google.com/drive/u/0/folders/1F3QOM7i1IWkM8bThJoqMSoWAxv-N2tD5
Лучшие веса находятся в папке https://drive.google.com/drive/folders/1B5TvAcy3l6usqY_hgWtUmWa4Il-hJAgL?usp=sharing
для того чтобы загрузить, нужно загрузить модель по имени весов, для osnet_x1_0.pth.tar-20 это osnet_x1_0 
и веса к ней osnet_x1_0.pth.tar-20, как такое выглядит

 model = torchreid.models.build_model(
        name='osnet_ain_x1_0', 
        num_classes=702, 
    )
    device = torch.device('cpu')
    model.to(device)
    torchreid.utils.load_pretrained_weights(model,
                                            r"C:\Users\Ektomo\PycharmProjects\pythonProject2\model\zoo\osnet_ain_x1_0.pth")
    model.eval()
    return model

Результаты экспериментов можно найти здесь: 
https://app.clear.ml/projects/6fff45728367482ab9970043665ffc97/experiments/1116ededd6aa4d1e9274cf66d9521e7a/output/execution
https://app.clear.ml/projects/6fff45728367482ab9970043665ffc97/experiments/f135de34f4b1445facdcd7f004475e87/output/execution

Таблица с результатами:

![image](https://github.com/AstuteVision/ReId/assets/70652745/671c296c-01d5-4329-9307-0f292c5fb3ef)



