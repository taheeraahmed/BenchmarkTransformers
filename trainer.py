from utils import MetricLogger, ProgressLogger
from models import ClassificationNet, build_classification_model
import time
import torch
from tqdm import tqdm


def train_one_epoch(data_loader_train, device, model, criterion, optimizer, epoch, accumulation_steps=1):
    batch_time = MetricLogger('Time', ':6.3f')
    losses = MetricLogger('Loss', ':.4e')
    progress = ProgressLogger(
        len(data_loader_train) // accumulation_steps,
        [batch_time, losses],
        prefix="Epoch: [{}]".format(epoch))

    model.train()

    optimizer.zero_grad()  # Initialize gradients to zero at the start of the epoch
    end = time.time()
    for i, (samples, targets) in enumerate(data_loader_train):
        samples, targets = samples.float().to(device), targets.float().to(device)
        outputs = model(samples)
        loss = criterion(outputs, targets) / accumulation_steps  # Normalize the loss to account for accumulation

        loss.backward()  # Accumulate gradients

        if (i + 1) % accumulation_steps == 0:
            optimizer.step()  # Update weights
            optimizer.zero_grad()  # Reset gradients

            # Only update loss and timer for batches where we perform an optimizer step
            losses.update(loss.item() * accumulation_steps, samples.size(0))  # Scale loss back up
            batch_time.update(time.time() - end)
            end = time.time()

            # Display progress every 50 optimizer steps (adjust as needed)
            if (i // accumulation_steps) % 50 == 0:
                progress.display(i // accumulation_steps)

    # If there are any remaining gradients that haven't been stepped through at the end of the epoch
    if len(data_loader_train) % accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()

def evaluate(data_loader_val, device, model, criterion):
  model.eval()

  with torch.no_grad():
    batch_time = MetricLogger('Time', ':6.3f')
    losses = MetricLogger('Loss', ':.4e')
    progress = ProgressLogger(
      len(data_loader_val),
      [batch_time, losses], prefix='Val: ')

    end = time.time()
    for i, (samples, targets) in enumerate(data_loader_val):
      samples, targets = samples.float().to(device), targets.float().to(device)

      outputs = model(samples)
      loss = criterion(outputs, targets)

      losses.update(loss.item(), samples.size(0))
      losses.update(loss.item(), samples.size(0))
      batch_time.update(time.time() - end)
      end = time.time()

      if i % 50 == 0:
        progress.display(i)

  return losses.avg


def test_classification(checkpoint, data_loader_test, device, args):
  model = build_classification_model(args)
  print(model)

  modelCheckpoint = torch.load(checkpoint)
  state_dict = modelCheckpoint['state_dict']
  for k in list(state_dict.keys()):
    if k.startswith('module.'):
      state_dict[k[len("module."):]] = state_dict[k]
      del state_dict[k]

  msg = model.load_state_dict(state_dict)
  assert len(msg.missing_keys) == 0
  print("=> loaded pre-trained model '{}'".format(checkpoint))

  if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)
  model.to(device)

  model.eval()

  y_test = torch.FloatTensor().cuda()
  p_test = torch.FloatTensor().cuda()

  with torch.no_grad():
    for i, (samples, targets) in enumerate(tqdm(data_loader_test)):
      targets = targets.cuda()
      y_test = torch.cat((y_test, targets), 0)

      if len(samples.size()) == 4:
        bs, c, h, w = samples.size()
        n_crops = 1
      elif len(samples.size()) == 5:
        bs, n_crops, c, h, w = samples.size()

      varInput = torch.autograd.Variable(samples.view(-1, c, h, w).cuda())

      out = model(varInput)
      if args.data_set == "RSNAPneumonia":
        out = torch.softmax(out,dim = 1)
      else:
        out = torch.sigmoid(out)
      outMean = out.view(bs, n_crops, -1).mean(1)
      p_test = torch.cat((p_test, outMean.data), 0)

  return y_test, p_test




