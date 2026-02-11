def extract_loss_history_qcpbc(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    val_losses = []
    for iline, line in enumerate(lines):
        if "Cycle" in line:
            start_iline = iline + 2
            break
    for iline in range(start_iline, len(lines)-1):
        line = lines[iline]
        value = float(line.strip().split()[1])
        val_losses.append(value)
    return val_losses