"""Extract the historical QCPBC optimization loss column from a text log."""


def extract_loss_history_qcpbc(filename):
    with open(filename) as handle:
        lines = handle.readlines()
    header = next((index for index, line in enumerate(lines) if "Cycle" in line), None)
    if header is None:
        raise ValueError("QCPBC log does not contain a Cycle table")
    losses = []
    for line in lines[header+2:]:
        fields = line.split()
        if not fields:
            continue
        try:
            int(fields[0])
            loss = float(fields[1])
        except (ValueError, IndexError):
            if losses:
                break
            continue
        losses.append(loss)
    return losses
