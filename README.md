# Обучение с подкреплением для траектории полного обхода динамической среды




### Run example

1. Install requirements:
`pip install -r requirements.txt`

2. Clone submodules
`git submodule update --remote`

3. Run training
`python train.py`

### Награда 
reward = reward - (info['step'] / self.max_steps) + (info['new_explored'] / info['step'])

Мы штрафуем за то, что прошло больше шагов (тк задача обойти за минимальное количество шагов)

Награждаем за новые исследования 
### Обучение

Сохраненная статистика: [wandb](https://wandb.ai/yumvolkova/ProdStory-Sensor/runs/7krriods?workspace=user-yumvolkova)

![plot](https://github.com/YUMVOLKOVA/ProdStory-Sensor/blob/main/gifs_179_a83c13bd630664dd33ca.gif)

В качестве алгоритма был выбрал ppo.
Судя по гифкам, агент часто останавливается и "зависает". Следует выбрать другую функцию награды.

### Другие запуски

[wandb](https://wandb.ai/yumvolkova/ProdStory-Sensor?workspace=user-yumvolkova)

Награда 

```
if info['moved']:
  reward = 0.5 * reward + (info['total_explored'] / info['total_cells']) - (info['step'] / self._max_steps)
else:
  reward = -1
```
