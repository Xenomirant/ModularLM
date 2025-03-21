# ModularLM

В данном репозитории хранится код для проекта "Локализации грамматики".

В рамках проекта:
- Подготовили данные: использовали синтетический полиперсональный русский язык, полученный добавлением дополнительных личных окончаний к глаголам, имеющим объект (тетрадка `polyagr_data_creation.ipynb`);
- Дообучили Берт на наших данных (тетрадка `mlm_pretrain.ipynb`);
- Провели следующие эксперименты:
    - Обучили модель флагом (полиперсональный язык или нет) и дополнительной частью для различения полиперсональности (тетрадки вида `modular_ml_???.ipynb`);
    - Обучили модель разносить тексты с полиаперсональностью и без в разные части пространства за счет косинусного лосса с инерцией (тетрадки вида `bert_tuning_???.ipynb`).
 
Результаты и более подробное описание были опубликованы в статье на Нейроинформатике
> Kudriashov S. et al. The more polypersonal the better-a short look on space geometry of fine-tuned layers //International Conference on Neuroinformatics. – Cham : Springer Nature Switzerland, 2024. – С. 13-22.
