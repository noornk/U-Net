
test = test_loader

i = 0
for a, b in test: # for X, y in dataloader:
  j = 0
  for j in range(0, len(a)):
    X = a[j].float()
    y = b[j].float()
    X = X.to(device)  # [N, 1, H, W]
    y = y.to(device)  # [N, H, W] with class indices (0, 1)
    y = y.squeeze(0)
    print(y.shape)

    plt.figure()
    plt.imshow(torch.exp(y[0,:,:]).detach().cpu())  # plot target
    plt.figure()
    plt.imshow((X[0,0,:,:]).detach().cpu())  # plot base image
    plt.figure()
    plt.imshow(torch.exp(prediction[0,0,:,:].detach().cpu()) ) # plot class0
    plt.figure()
    plt.imshow(torch.exp(prediction[0,1,:,:].detach().cpu())) # plot class1

    

  i = i + 1
