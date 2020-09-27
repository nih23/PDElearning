import torch
import numpy as np
import matplotlib.pyplot as plt
import pathlib

def writeIntermediateState(timeStep, model, dataset, epoch, fileWriter, csystem, identifier="PDE"):
    """
    Functions that write intermediate solutions to tensorboard
    """
    if fileWriter is None:
        return

    nx = csystem['nx']
    ny = csystem['nx']

    x, y, t = dataset.getInput(timeStep, csystem)
    x = torch.Tensor(x).float().cuda()
    y = torch.Tensor(y).float().cuda()
    t = torch.Tensor(t).float().cuda()

    inputX = torch.stack([x, y, t], 1)
    UV = model.forward(inputX).detach().cpu().numpy()

    u = UV[:, 0].reshape((nx, ny))
    v = UV[:, 1].reshape((nx, ny))

    h = u ** 2 + v ** 2

    fig = plt.figure()
    plt.imshow(u, extent=[-3,3,-3,3], cmap='jet')
    plt.colorbar()
    fileWriter.add_figure('%s-real/t%.2f' %
                          (identifier, t[0].cpu().numpy()), fig, epoch)
    plt.close(fig)

    fig = plt.figure()
    plt.imshow(v, extent=[-3,3,-3,3], cmap='jet')
    plt.colorbar()
    fileWriter.add_figure('%s-imag/t%.2f' %
                          (identifier, t[0].cpu().numpy()), fig, epoch)
    plt.close(fig)

    fig = plt.figure()
    plt.imshow(h, extent=[-3,3,-3,3], cmap='jet')
    plt.colorbar()
    fileWriter.add_figure('%s-norm/t%.2f' %
                          (identifier, t[0].cpu().numpy()), fig, epoch)
    plt.close(fig)


def valLoss(model, dataset, timeStep, csystem):
    x, y, t = dataset.getInput(timeStep, csystem)
    x = torch.Tensor(x).float().cuda()
    y = torch.Tensor(y).float().cuda()
    t = torch.Tensor(t).float().cuda()

    inputX = torch.stack([x, y, t], 1)
    UV = model.forward(inputX).detach().cpu().numpy()
    uPred = UV[:, 0].reshape(-1)
    vPred = UV[:, 1].reshape(-1)

    # load label data
    uVal, vVal = dataset.getFrame(timeStep, csystem)
    uVal = np.array(uVal).reshape(-1)
    vVal = np.array(vVal).reshape(-1)

    valLoss_u = np.max(abs(uVal - uPred))
    valLoss_v = np.max(abs(vVal - vPred))
    valSqLoss_u = np.sqrt(np.sum(np.power(uVal - uPred, 2)))
    valSqLoss_v = np.sqrt(np.sum(np.power(vVal - vPred, 2)))

    return valLoss_u, valLoss_v, valSqLoss_u, valSqLoss_v


def writeValidationLoss(timeStep, model, dataset, epoch, writer, csystem, identifier):
    if writer is None:
        return

    _, _, t = dataset.getInput(timeStep, csystem)
    t = torch.Tensor(t).float().cuda()

    valLoss_u, valLoss_v, valSqLoss_u, valSqLoss_v = valLoss(
        model, dataset, timeStep, csystem)
    valLoss_uv = valLoss_u + valLoss_v
    valSqLoss_uv = valSqLoss_u + valSqLoss_v
    writer.add_scalar("inf: L_%s/u/t%.2f" %
                      (identifier, t[0].cpu().numpy()), valLoss_u, epoch)
    writer.add_scalar("inf: L_%s/v/t%.2f" %
                      (identifier, t[0].cpu().numpy()), valLoss_v, epoch)
    writer.add_scalar("inf: L_%s/uv/t%.2f" %
                      (identifier, t[0].cpu().numpy()), valLoss_uv, epoch)
    writer.add_scalar("2nd: L_%s/u/t%.2f" %
                      (identifier, t[0].cpu().numpy()), valSqLoss_u, epoch)
    writer.add_scalar("2nd: L_%s/v/t%.2f" %
                      (identifier, t[0].cpu().numpy()), valSqLoss_v, epoch)
    writer.add_scalar("2nd: L_%s/uv/t%.2f" %
                      (identifier, t[0].cpu().numpy()), valSqLoss_uv, epoch)


def save_checkpoint(model, path, epoch):
    # print(model.state_dict().keys())
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    state = {
        'model': model.state_dict(),
    }
    torch.save(state, path + 'model_' + str(epoch)+'.pt')
    print("saving model to ---> %s" % (path + 'model_' + str(epoch)+'.pt'))


def load_checkpoint(model, path):
    device = torch.device('cpu')
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model'])


def getDefaults():
    # static parameter
    nx = 200
    ny = 200
    nt = 1000
    xmin = -3
    xmax = 3
    ymin = -3
    ymax = 3
    dt = 0.001
    tmax = 1
    numOfEnergySamplingPointsX = 100
    numOfEnergySamplingPointsY = 100

    coordinateSystem = {"x_lb": xmin, "x_ub": xmax, "y_lb": ymin,
                        "y_ub": ymax, "nx": nx, "ny": ny, "nt": nt, "dt": dt}

    return coordinateSystem, numOfEnergySamplingPointsX, numOfEnergySamplingPointsY, tmax


def getVariables(model, dataset, cSystem, time, step=1):

    x, y, t = dataset.getInput(time, cSystem, step)

    x = torch.Tensor(x).float().cuda()
    y = torch.Tensor(y).float().cuda()
    t = torch.Tensor(t).float().cuda()

    u, v, u_yy, v_yy, u_xx, v_xx, u_t, v_t = model.net_uv(x, y, t)

    x = x.view(-1)
    y = y.view(-1)

    X = torch.stack([x, y, u, v, u_yy, v_yy, u_xx, v_xx], 1)

    f = -model.forward_hpm(X)

    dudt = f[:, 0]
    dvdt = f[:, 1]

    dudt = dudt.cpu().detach().numpy().reshape(-1, 1)
    dvdt = dvdt.cpu().detach().numpy().reshape(-1, 1)

    x = x.cpu().detach().numpy().reshape(-1)
    y = y.cpu().detach().numpy().reshape(-1)
    u = u.cpu().detach().numpy().reshape(-1)
    v = v.cpu().detach().numpy().reshape(-1)
    u_xx = u_xx.cpu().detach().numpy().reshape(-1)
    u_yy = u_yy.cpu().detach().numpy().reshape(-1)
    v_xx = v_xx.cpu().detach().numpy().reshape(-1)
    v_yy = v_yy.cpu().detach().numpy().reshape(-1)

    return x, y, u, v, u_xx, u_yy, v_xx, v_yy, dudt, dvdt


def write_coefficients(model, dataset, epoch, cSystem, time, fileWriter):


    x, y, u, v, u_xx, u_yy, v_xx, v_yy, dudt, dvdt = getVariables(
    model, dataset, cSystem, time)

    l1 = []
    l2 = []

    for i in range(cSystem['nx']*cSystem['ny']):

        if x[i] == 0 and y[i] == 0:
            c1 = 0
            c2 = 0
        else:
            a = np.array([[-v_xx[i]-v_yy[i], u_xx[i]+u_yy[i]], [v[i]
                                                                * x[i]**2 + v[i]*y[i]**2, -u[i]*x[i]**2 - u[i]*y[i]**2]])
            b = np.array([dudt[i], dvdt[i]])
            c1, c2 = np.linalg.solve(a, b).reshape(-1)

        if c1 > 10:
            c1 = 10
        elif c1 < -10:
            c1 = -10
        if c2 > 10:
            c2 = 10
        elif c2 < -10:
            c2 = -10

        l1.append(c1)
        l2.append(c2)

    fileWriter.add_scalar("median(C1)", np.median(l1), epoch)
    fileWriter.add_scalar("median(C2)", np.median(l2), epoch)

    fileWriter.add_scalar("var(C1)", np.var(l1), epoch)
    fileWriter.add_scalar("var(C2)", np.var(l2), epoch)

    l1 = np.array(l1).reshape((cSystem['nx'], cSystem['ny']))
    fig = plt.figure()
    plt.imshow(l1, extent=[-3,3,-3,3], cmap='jet')
    plt.colorbar()
    fileWriter.add_figure('c1/t%.2f' % (time), fig, epoch)
    plt.close(fig)

    l2 = np.array(l2).reshape((cSystem['nx'], cSystem['ny']))
    fig = plt.figure()
    plt.imshow(l2, extent=[-3,3,-3,3], cmap='jet')
    plt.colorbar()
    fileWriter.add_figure('c2/t%.2f' % (time), fig, epoch)
    plt.close(fig)

    return 0


def write_dependencies(model, dataset, epoch, fileWriter, cSystem, time=0):

    x, y, u, v, u_xx, u_yy, v_xx, v_yy, dudt, dvdt = getVariables(
        model, dataset, cSystem, time)

    fig = plt.figure()
    plt.scatter(v_xx, dudt)
    plt.xlabel('v_xx')
    plt.ylabel('du/dt')
    fileWriter.add_figure('dudt ~ v_xx', fig, epoch)
    plt.close(fig)

    fig = plt.figure()
    plt.scatter(v_yy, dudt)
    plt.xlabel('v_yy')
    plt.ylabel('du/dt')
    fileWriter.add_figure('dudt ~ v_yy', fig, epoch)
    plt.close(fig)

    fig = plt.figure()
    plt.scatter(v*(x**2), dudt)
    plt.xlabel('v * x^2')
    plt.ylabel('du/dt')
    fileWriter.add_figure('dudt ~ v*x^2', fig, epoch)
    plt.close(fig)

    fig = plt.figure()
    plt.scatter(v*(y**2), dudt)
    plt.xlabel('v * y^2')
    plt.ylabel('du/dt')
    fileWriter.add_figure('dudt ~ v*y^2', fig, epoch)
    plt.close(fig)

    fig = plt.figure()
    plt.scatter(u_xx, dvdt)
    plt.xlabel('u_xx')
    plt.ylabel('dv/dt')
    fileWriter.add_figure('dvdt ~ u_xx', fig, epoch)
    plt.close(fig)

    fig = plt.figure()
    plt.scatter(u_yy, dvdt)
    plt.xlabel('u_yy')
    plt.ylabel('dv/dt')
    fileWriter.add_figure('dvdt ~ u_yy', fig, epoch)
    plt.close(fig)

    fig = plt.figure()
    plt.scatter(u*(x**2), dudt)
    plt.xlabel('u * x^2')
    plt.ylabel('dv/dt')
    fileWriter.add_figure('dvdt ~ u*x^2', fig, epoch)
    plt.close(fig)

    fig = plt.figure()
    plt.scatter(u*(y**2), dvdt)
    plt.xlabel('u * y^2')
    plt.ylabel('dv/dt')
    fileWriter.add_figure('dvdt ~ u*y^2', fig, epoch)
    plt.close(fig)

    return 0
