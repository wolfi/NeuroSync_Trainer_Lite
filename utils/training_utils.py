# training_utils.py

import torch.nn as nn
import torch
import time
import matplotlib.pyplot as plt
import os

def init_weights(m):
    if isinstance(m, (nn.Linear, nn.Conv1d)):
        print(f"Initializing {m} with normal distribution")
        nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

def count_parameters(model):
    """Count and print the number of parameters in a model."""
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {param_count}")
    return param_count

def calculate_gradient_norm(model):
    """Calculate and return the gradient norm for the model."""
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm

def print_training_progress(batch_idx, total_norm, batch_loss, batch_step, epoch, total_epochs, dataloader_len, pbar):
    """Print training progress and update the progress bar."""
    print(f"Batch {batch_idx}, Gradient Norm: {total_norm}")
    if pbar is not None:
        pbar.update(1)
    print(f"Step [{batch_step}/{pbar.total}], Epoch [{epoch + 1}/{total_epochs}], Batch [{batch_idx + 1}/{dataloader_len}], Current Loss: {batch_loss:.4f}")

def print_epoch_summary(epoch, total_epochs, epoch_loss, dataloader_len, epoch_time):
    """Print the summary of the epoch."""
    print(f"Epoch [{epoch + 1}/{total_epochs}], Loss: {epoch_loss / dataloader_len:.4f}, Time: {epoch_time:.2f} seconds")

def save_gradient_norm_plot(epoch, gradient_norms, save_dir):
    """Save a plot of gradient norms over the batches in an epoch."""
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(gradient_norms, label="Gradient Norm")
    plt.xlabel("Batch Index")
    plt.ylabel("Gradient Norm")
    plt.title(f"Gradient Norm Fluctuations (Epoch {epoch + 1})")
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(save_dir, f"gradient_norms_epoch_{epoch + 1}.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Gradient norm plot saved to {plot_path}")




def train_one_epoch(epoch, model, dataloader, criterion, optimizer, device, clip, batch_step=0, pbar=None, total_epochs=None):
    model.train()
    epoch_loss = 0
    start_time = time.time()

    total_steps = total_epochs * len(dataloader)  # Total training steps
    gradient_norms = []

    for batch_idx, (src, trg) in enumerate(dataloader):
        src, trg = src.to(device), trg.to(device)

        optimizer.zero_grad()
        output = model(src)
        current_step = batch_step + (epoch * len(dataloader)) + batch_idx
        loss = criterion(output, trg, current_step=current_step, total_steps=total_steps)
        loss.backward()

        total_norm = calculate_gradient_norm(model)
        print_training_progress(batch_idx, total_norm, loss.item(), batch_step, epoch, total_epochs, len(dataloader), pbar)

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        gradient_norms.append(total_norm)
        epoch_loss += loss.item()
        batch_step += 1

    end_time = time.time()
    print_epoch_summary(epoch, total_epochs, epoch_loss, len(dataloader), end_time - start_time)

    save_gradient_norm_plot(epoch, gradient_norms, save_dir="dataset/validation_plots/gradient_norms")

    return batch_step


### multi GPU testing 
def train_one_epoch_multi_gpu_2(
    epoch, model_0, model_1, dataloader, criterion, optimizer,
    device0, device1, clip, batch_step=0, pbar=None, total_epochs=None
):
    model_0.train()
    model_1.train()

    epoch_loss = 0
    start_time = time.time()
    
    steps_per_epoch = len(dataloader) // 2
    total_steps = total_epochs * steps_per_epoch
    gradient_norms = []

    data_iter = iter(dataloader)

    for step_idx in range(steps_per_epoch):
        step_start = time.time()

        # 1. Fetch two batches
        try:
            src_0, trg_0 = next(data_iter)
            src_1, trg_1 = next(data_iter)
        except StopIteration:
            print(f"Dropping leftover mini-batches at step {step_idx}.")
            break

        # 2. Move batches to their respective GPUs
        src_0, trg_0 = src_0.to(device0, non_blocking=True), trg_0.to(device0, non_blocking=True)
        src_1, trg_1 = src_1.to(device1, non_blocking=True), trg_1.to(device1, non_blocking=True)

        # 3. Zero only model_0's gradients before computing new ones
        optimizer.zero_grad()  # ✅ Correctly zero only model_0

        # 4. Compute current step
        current_step = batch_step + (epoch * steps_per_epoch) + step_idx

        # 5. Forward/backward passes on each GPU model
        output_0 = model_0(src_0)
        loss_0 = criterion(output_0, trg_0, current_step=current_step, total_steps=total_steps)
        loss_0.backward()

        output_1 = model_1(src_1)
        loss_1 = criterion(output_1, trg_1, current_step=current_step, total_steps=total_steps)
        loss_1.backward()

        # 6. Synchronize GPUs to ensure all operations complete before averaging gradients
        torch.cuda.synchronize(device0)
        torch.cuda.synchronize(device1)

        # 7. Gradient Averaging BEFORE Zeroing
        grad_0_list = []
        grad_1_list = []
        valid_params = []

        for p0, p1 in zip(model_0.parameters(), model_1.parameters()):
            if p0.grad is not None and p1.grad is not None:
                grad_0_list.append(p0.grad.data.view(-1))
                grad_1_list.append(p1.grad.data.view(-1))
                valid_params.append((p0, p1))

        if len(grad_0_list) > 0:
            grad_0_flat = torch.cat(grad_0_list)
            grad_1_flat = torch.cat(grad_1_list).to(device0)
            grad_0_flat.add_(grad_1_flat).div_(2.0)

            offset = 0
            for (p0, p1) in valid_params:
                length = p0.numel()
                p0.grad.data.copy_(grad_0_flat[offset:offset+length].view_as(p0))
                offset += length

        # 8. Synchronize again to ensure gradient averaging operations completed
        torch.cuda.synchronize(device0)
        torch.cuda.synchronize(device1)

        # 9. Compute gradient norm before clipping
        raw_total_norm = calculate_gradient_norm(model_0)
        gradient_norms.append(raw_total_norm)

        # 10. Clip gradients, optimizer step, and sync models
        torch.nn.utils.clip_grad_norm_(model_0.parameters(), clip)
        optimizer.step()

        # 11. Sync updated parameters from model_0 to model_1
        for param_0, param_1 in zip(model_0.parameters(), model_1.parameters()):
            param_1.data.copy_(param_0.data.to(device1))

        # 12. ✅ Reset model_1's gradients after updating model_0
        for p in model_1.parameters():
            if p.grad is not None:
                p.grad.zero_()

        # 13. Compute loss info
        batch_loss = (loss_0.item() + loss_1.item()) / 2.0
        epoch_loss += batch_loss

        # Print step info
        print(f"Step Index [{step_idx+1}/{steps_per_epoch}], Loss: {batch_loss:.4f}, Grad Norm (pre-clip): {raw_total_norm:.4f}")

        if pbar is not None:
            pbar.update(1)

        batch_step += 1

    end_time = time.time()
    print_epoch_summary(epoch, total_epochs, epoch_loss, steps_per_epoch, end_time - start_time)

    # Save the gradient norm plot for the entire epoch
    save_gradient_norm_plot(
        epoch, gradient_norms,
        save_dir="dataset/validation_plots/gradient_norms"
    )

    return batch_step

def train_one_epoch_multi_gpu_3(
    epoch, model_0, model_1, model_2, dataloader, criterion, optimizer,
    device0, device1, device2, clip, batch_step=0, pbar=None, total_epochs=None
):
    model_0.train()
    model_1.train()
    model_2.train()

    epoch_loss = 0
    start_time = time.time()
    steps_per_epoch = len(dataloader) // 3
    total_steps = total_epochs * steps_per_epoch
    gradient_norms = []

    data_iter = iter(dataloader)

    for step_idx in range(steps_per_epoch):
        step_start = time.time()

        # 1. Fetch three batches
        try:
            src_0, trg_0 = next(data_iter)
            src_1, trg_1 = next(data_iter)
            src_2, trg_2 = next(data_iter)
        except StopIteration:
            print(f"Dropping leftover mini-batches at step {step_idx}.")
            break

        # 2. Move to GPUs
        src_0, trg_0 = src_0.to(device0, non_blocking=True), trg_0.to(device0, non_blocking=True)
        src_1, trg_1 = src_1.to(device1, non_blocking=True), trg_1.to(device1, non_blocking=True)
        src_2, trg_2 = src_2.to(device2, non_blocking=True), trg_2.to(device2, non_blocking=True)

        # 3. Zero gradients for model_0 only
        optimizer.zero_grad()

        # 4. Compute current step
        current_step = batch_step + (epoch * steps_per_epoch) + step_idx

        # 5. Forward/backward passes
        output_0 = model_0(src_0)
        loss_0 = criterion(output_0, trg_0, current_step=current_step, total_steps=total_steps)
        loss_0.backward()

        output_1 = model_1(src_1)
        loss_1 = criterion(output_1, trg_1, current_step=current_step, total_steps=total_steps)
        loss_1.backward()

        output_2 = model_2(src_2)
        loss_2 = criterion(output_2, trg_2, current_step=current_step, total_steps=total_steps)
        loss_2.backward()

        # 6. Synchronize GPUs
        torch.cuda.synchronize(device0)
        torch.cuda.synchronize(device1)
        torch.cuda.synchronize(device2)

        # 7. Gradient Averaging BEFORE Zeroing
        grad_0_list, grad_1_list, grad_2_list = [], [], []
        valid_params = []

        for p0, p1, p2 in zip(model_0.parameters(), model_1.parameters(), model_2.parameters()):
            if p0.grad is not None and p1.grad is not None and p2.grad is not None:
                grad_0_list.append(p0.grad.view(-1))
                grad_1_list.append(p1.grad.view(-1))
                grad_2_list.append(p2.grad.view(-1))
                valid_params.append((p0, p1, p2))

        if grad_0_list:
            grad_0_flat = torch.cat(grad_0_list)
            grad_1_flat = torch.cat(grad_1_list).to(device0)
            grad_2_flat = torch.cat(grad_2_list).to(device0)

            grad_0_flat.add_(grad_1_flat).add_(grad_2_flat).div_(3.0)

            offset = 0
            for p0, p1, p2 in valid_params:
                length = p0.numel()
                p0.grad.copy_(grad_0_flat[offset:offset + length].view_as(p0))
                offset += length

        torch.cuda.synchronize(device0)

        # 8. Clip gradients and update model_0
        torch.nn.utils.clip_grad_norm_(model_0.parameters(), clip)
        optimizer.step()

        # 9. Sync updated parameters
        for param_0, param_1, param_2 in zip(model_0.parameters(), model_1.parameters(), model_2.parameters()):
            param_1.data.copy_(param_0.data.to(device1))
            param_2.data.copy_(param_0.data.to(device2))

        # 10. Zero gradients for model_1 and model_2 AFTER updating
        for p in model_1.parameters():
            if p.grad is not None:
                p.grad.zero_()
        for p in model_2.parameters():
            if p.grad is not None:
                p.grad.zero_()

        # 11. Compute loss
        batch_loss = (loss_0.item() + loss_1.item() + loss_2.item()) / 3.0
        epoch_loss += batch_loss
        gradient_norms.append(calculate_gradient_norm(model_0))

        print(f"Step [{step_idx+1}/{steps_per_epoch}], Loss: {batch_loss:.4f}")

        if pbar is not None:
            pbar.update(1)

        batch_step += 1

    print_epoch_summary(epoch, total_epochs, epoch_loss, steps_per_epoch, time.time() - start_time)
    save_gradient_norm_plot(epoch, gradient_norms, save_dir="dataset/validation_plots/gradient_norms")

    return batch_step



def train_one_epoch_multi_gpu_4(
    epoch,
    model_0,
    model_1,
    model_2,
    model_3,
    dataloader,
    criterion,
    optimizer,
    device0,
    device1,
    device2,
    device3,
    clip,
    batch_step=0,
    pbar=None,
    total_epochs=None
):
    """
    4-GPU training routine where each GPU processes one mini-batch.
    We vectorize the gradient averaging on device0 (flatten & sum).
    If we can't fetch exactly 4 batches at any step, we break out
    to effectively 'drop' the leftover.
    """
    model_0.train()
    model_1.train()
    model_2.train()
    model_3.train()

    epoch_loss = 0
    start_time = time.time()
    
    steps_per_epoch = len(dataloader) // 4
    total_steps = total_epochs * steps_per_epoch
    gradient_norms = []
    
    data_iter = iter(dataloader)

    for step_idx in range(steps_per_epoch):
        step_start = time.time()

        # 1. Fetch four batches
        try:
            src_0, trg_0 = next(data_iter)
            src_1, trg_1 = next(data_iter)
            src_2, trg_2 = next(data_iter)
            src_3, trg_3 = next(data_iter)
        except StopIteration:
            print(f"Dropping leftover mini-batches at step {step_idx}.")
            break

        # 2. Move batches to respective GPUs
        src_0, trg_0 = src_0.to(device0, non_blocking=True), trg_0.to(device0, non_blocking=True)
        src_1, trg_1 = src_1.to(device1, non_blocking=True), trg_1.to(device1, non_blocking=True)
        src_2, trg_2 = src_2.to(device2, non_blocking=True), trg_2.to(device2, non_blocking=True)
        src_3, trg_3 = src_3.to(device3, non_blocking=True), trg_3.to(device3, non_blocking=True)

        # 3. Zero gradients only for model_0
        optimizer.zero_grad()

        # 4. Compute current step
        current_step = batch_step + (epoch * steps_per_epoch) + step_idx

        # 5. Forward/backward passes
        output_0 = model_0(src_0)
        loss_0 = criterion(output_0, trg_0, current_step=current_step, total_steps=total_steps)
        loss_0.backward()

        output_1 = model_1(src_1)
        loss_1 = criterion(output_1, trg_1, current_step=current_step, total_steps=total_steps)
        loss_1.backward()

        output_2 = model_2(src_2)
        loss_2 = criterion(output_2, trg_2, current_step=current_step, total_steps=total_steps)
        loss_2.backward()

        output_3 = model_3(src_3)
        loss_3 = criterion(output_3, trg_3, current_step=current_step, total_steps=total_steps)
        loss_3.backward()

        # 6. Synchronize GPUs
        torch.cuda.synchronize(device0)
        torch.cuda.synchronize(device1)
        torch.cuda.synchronize(device2)
        torch.cuda.synchronize(device3)

        # 7. Gradient Averaging BEFORE Zeroing
        grad_0_list, grad_1_list, grad_2_list, grad_3_list = [], [], [], []
        valid_params = []

        for p0, p1, p2, p3 in zip(
            model_0.parameters(),
            model_1.parameters(),
            model_2.parameters(),
            model_3.parameters()
        ):
            if p0.grad is not None and p1.grad is not None and p2.grad is not None and p3.grad is not None:
                grad_0_list.append(p0.grad.view(-1))
                grad_1_list.append(p1.grad.view(-1))
                grad_2_list.append(p2.grad.view(-1))
                grad_3_list.append(p3.grad.view(-1))
                valid_params.append((p0, p1, p2, p3))

        if grad_0_list:
            grad_0_flat = torch.cat(grad_0_list)
            grad_1_flat = torch.cat(grad_1_list).to(device0)
            grad_2_flat = torch.cat(grad_2_list).to(device0)
            grad_3_flat = torch.cat(grad_3_list).to(device0)

            grad_0_flat.add_(grad_1_flat).add_(grad_2_flat).add_(grad_3_flat).div_(4.0)

            offset = 0
            for p0, p1, p2, p3 in valid_params:
                length = p0.numel()
                p0.grad.copy_(grad_0_flat[offset:offset + length].view_as(p0))
                offset += length

        torch.cuda.synchronize(device0)

        # 8. Clip gradients and update model_0
        torch.nn.utils.clip_grad_norm_(model_0.parameters(), clip)
        optimizer.step()

        # 9. Sync updated parameters from model_0 to model_1, model_2, model_3
        for param_0, param_1, param_2, param_3 in zip(
            model_0.parameters(),
            model_1.parameters(),
            model_2.parameters(),
            model_3.parameters()
        ):
            param_1.data.copy_(param_0.data.to(device1))
            param_2.data.copy_(param_0.data.to(device2))
            param_3.data.copy_(param_0.data.to(device3))

        # 10. Zero gradients for model_1, model_2, model_3 AFTER updating
        for p in model_1.parameters():
            if p.grad is not None:
                p.grad.zero_()
        for p in model_2.parameters():
            if p.grad is not None:
                p.grad.zero_()
        for p in model_3.parameters():
            if p.grad is not None:
                p.grad.zero_()

        # 11. Compute loss
        batch_loss = (loss_0.item() + loss_1.item() + loss_2.item() + loss_3.item()) / 4.0
        epoch_loss += batch_loss
        gradient_norms.append(calculate_gradient_norm(model_0))

        # 12. Print step info
        print(f"Step [{step_idx+1}/{steps_per_epoch}], Loss: {batch_loss:.4f}")

        if pbar is not None:
            pbar.update(1)

        batch_step += 1

    print_epoch_summary(epoch, total_epochs, epoch_loss, steps_per_epoch, time.time() - start_time)
    save_gradient_norm_plot(epoch, gradient_norms, save_dir="dataset/validation_plots/gradient_norms")

    return batch_step




def flatten_grads(model):
    """
    Flatten all .grad tensors from 'model' into a single 1D tensor.
    If a param has .grad=None, we either fill with zeros or skip it.
    Here we fill with zeros to preserve alignment, so all models have the same flattened size.
    """
    grads = []
    for p in model.parameters():
        if p.grad is None:
            # Allocate zeros of same shape, device, dtype
            # so that the flattened vector sizes match across all GPUs
            z = torch.zeros(p.numel(), device=p.device, dtype=p.dtype)
            grads.append(z)
        else:
            grads.append(p.grad.view(-1))
    return torch.cat(grads)

def unflatten_grads(flat, model):
    """
    Copy slices from 'flat' 1D tensor back into each param's .grad in 'model'.
    The param order & shapes must match exactly the loop in flatten_grads().
    """
    offset = 0
    for p in model.parameters():
        length = p.numel()
        if p.grad is None:
            # If you truly want to restore "None", just skip. 
            # Or create a zero-filled .grad in p if desired.
            offset += length
            continue
        p.grad.data.copy_(flat[offset:offset+length].view_as(p))
        offset += length



def validate_model(model, val_dataloader, criterion, device):
    model.eval()
    val_loss = 0
    total_steps = len(val_dataloader)  # Dummy total_steps for validation
    with torch.no_grad():
        for current_step, (src, trg) in enumerate(val_dataloader):
            src, trg = src.to(device), trg.to(device)
            output = model(src)
            # Pass dummy current_step and total_steps
            loss = criterion(output, trg, current_step, total_steps)
            val_loss += loss.item()
    return val_loss / len(val_dataloader)
