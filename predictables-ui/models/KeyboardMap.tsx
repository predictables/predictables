//
const KeyboardMap = (
  event: KeyboardEvent | any = null,
  isDrawerOpen: boolean = false,
  openDrawer: () => void,
  closeDrawer: () => void,
) => {
  // Function to be returned - scoped to the if statements below
  let func = () => {};

  if (event.key.toLowerCase() === 'a') {
    // not assigned yet
  }

  if (event.key.toLowerCase() === 'b') {
    // not assigned yet
  }

  if (event.key.toLowerCase() === 'c') {
    // not assigned yet
  }

  if (event.key.toLowerCase() === 'd') {
    // D - depends on context -> toggles drawer open/close
    if (isDrawerOpen) {
      func = closeDrawer;
    } else {
      func = openDrawer;
    }
  }

  if (event.key.toLowerCase() === 'e') {
    // not assigned yet
  }

  if (event.key.toLowerCase() === 'f') {
    // not assigned yet
  }

  if (event.key.toLowerCase() === 'g') {
    // not assigned yet
  }

  if (event.key.toLowerCase() === 'h') {
    // not assigned yet
  }

  if (event.key.toLowerCase() === 'i') {
    // not assigned yet
  }

  if (event.key.toLowerCase() === 'j') {
    // not assigned yet
  }

  if (event.key.toLowerCase() === 'k') {
    // not assigned yet
  }

  if (event.key.toLowerCase() === 'l') {
    // not assigned yet
  }

  if (event.key.toLowerCase() === 'm') {
    // not assigned yet
  }

  if (event.key.toLowerCase() === 'n') {
    // not assigned yet
  }

  if (event.key.toLowerCase() === 'o') {
    // not assigned yet
  }

  if (event.key.toLowerCase() === 'p') {
    // not assigned yet
  }

  if (event.key.toLowerCase() === 'q') {
    // not assigned yet
  }

  if (event.key.toLowerCase() === 'r') {
    // not assigned yet
  }

  if (event.key.toLowerCase() === 's') {
    // not assigned yet
  }

  if (event.key.toLowerCase() === 't') {
    // not assigned yet
  }

  if (event.key.toLowerCase() === 'u') {
    // not assigned yet
  }

  if (event.key.toLowerCase() === 'v') {
    // not assigned yet
  }

  if (event.key.toLowerCase() === 'w') {
    // not assigned yet
  }

  if (event.key.toLowerCase() === 'x') {
    // not assigned yet
  }

  if (event.key.toLowerCase() === 'y') {
    // not assigned yet
  }

  if (event.key.toLowerCase() === 'z') {
    // not assigned yet
  }

  if (event.key === 'Escape') {
    if (isDrawerOpen) {
      func = closeDrawer;
    }
  }

  return func;
};

export default KeyboardMap;
