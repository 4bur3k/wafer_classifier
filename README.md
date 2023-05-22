# Wafer Classifier
## О проекте
Проект для выпускной работы в НИТУ МИСиС. В *nb.ipynb* я подготавливаю данные и подготоавливаю для подачи в модели. Затем я сравниваю резульаты ResNet50 и CNN. 
В *main.py* реализован интерфейс для дмонстрации результатов.

 ## Как зпустить
 * Скачиваем [LSWMD.plk ](https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map), кладем в папку data. 
 
  * ```bash
    pip install -r requirements.txt
    ```
 * С помощью ноутбука делим данные на трейн и тест, сохраняем их в data/train.plk и data/test.plk
 * ```bash
   streamlit run main.py
   ``` 

# Результат 
 <img src="https://i.imgur.com/8mOtwcl.png" width="800">
 
 ### Я обязательно опишу результаты, но позже 🤥...
 
