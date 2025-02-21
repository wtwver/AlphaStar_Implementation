max_episodes = 5000

with tqdm.trange(max_episodes) as t:
  for i in t:
    loss = train_supervised_step(model, optimizer)
    if i % 100 == 0:
      model.save("LunaLander_SL_Model")
      print("loss: ", loss)