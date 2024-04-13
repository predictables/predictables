"""Train an embedding model for NAICS codes."""

from predictables.naics_cd_embedding import NAICSDefaults, NAICSDataLoader


def main() -> None:
    # Load the default configuration
    config, model, criterion, optimizer = NAICSDefaults().get()

    # Load the data
    data_loader = NAICSDataLoader()

    # Example training loop
    for epoch in range(10):
        for naics_codes, targets in data_loader:
            optimizer.zero_grad()
            probabilities = model(
                naics_codes[0], naics_codes[1], naics_codes[2], naics_codes[3], naics_codes[4]
            )
            loss = criterion(probabilities.squeeze(), targets)
            loss.backward()
            optimizer.step()

            print(f"Epoch {epoch}, Loss: {loss.item()}")

            