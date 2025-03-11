max_episodes = 5000

# Assuming tqdm is imported
with tqdm.trange(max_episodes) as t:
    for i in t:
        loss = train_supervised_step(model, optimizer)
        if i % 100 == 0:
            # PyTorch model saving
            torch.save(model.state_dict(), "LunaLander_SL_Model.pth")
            # Convert loss to Python scalar for printing
            print("loss: ", loss.item())