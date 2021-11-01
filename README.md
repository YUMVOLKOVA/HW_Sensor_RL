# Обучение с подкреплением для траектории полного обхода динамической среды




### Run example

1. Install requirements:
`pip install -r requirements.txt`

2. Clone submodules
`git submodule update --remote`

3. Run training
`python ppo_example.py`

### Награда 
обычная награда - штраф на количество шагов (текущий шаг/максимальное количетсво шагов) + награда за новые ячейки(новые ячейки / текущий шаг)

### Обучение

Для сохранений результатов: wandb

### Результат
