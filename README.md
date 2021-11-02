# Обучение с подкреплением для траектории полного обхода динамической среды




### Run example

1. Install requirements:
`pip install -r requirements.txt`

2. Clone submodules
`git submodule update --remote`

3. Run training
`python ppo_example.py`

### Награда 
reward = reward - (info['step'] / self.max_steps) + (info['new_explored'] / info['step'])
Мы штрафуем за то, что прошло больше шагов (тк задача обойти за минимальное количество шагов)
Награждаем за новые исследования 
### Обучение

Сохраненная статистика: [wandb](https://wandb.ai/yumvolkova/ProdStory-Sensor/runs/7krriods?workspace=user-yumvolkova)

![](https://storage.googleapis.com/wandb-production.appspot.com/yumvolkova/ProdStory-Sensor/7krriods/media/videos/gifs_179_a83c13bd630664dd33ca.gif?Expires=1635842317&GoogleAccessId=wandb-production%40appspot.gserviceaccount.com&Signature=KwKK2p0aQN2Tf3INGW%2FIM1eRkn5UtjkKWqO%2BspWnJtAkodw2D%2BHSataLjufsB%2FuxWSt52AnOIo1OVctEKbP1qqSjOK4tt6zHiTVkLOWCl5QUy1Uw0llXE8%2B4UgyEGiSU3sA%2BBiabygtUMCKcDCwWQ662xGL9d8HRCA9npEdieTbQ4rcxMuD6TpFu3uqkFfJqUpLk32sWbFWxgtLl72a5D8WEOyGNrRg17xB6nAmvh7%2BYt3qIfD%2BiDuRctIwqQo3tP6SZHiXfYebOObzUCzC6E94lGVuYlD2x89HVicIkpdWoEmjpEZH6C7WxWaY9ya9B4rMaF7oKztptL9eCxWDRLw%3D%3D)

В качестве алгоритма был выбрал ppo.
Эксперимент не получился. Судя по гифкам, агент часто останавливается и "зависает". Следует выбрать другую функцию награды.

