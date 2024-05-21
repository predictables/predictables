import { render, fireEvent } from '@testing-library/react';
import Navbar from './Navbar.component';

describe('Navbar Component', () => {
  test('renders the Logo component', () => {
    const { getByText } = render(<Navbar />);
    expect(getByText('Logo')).toBeInTheDocument();  // Assuming Logo renders a text "Logo"
  });

  test('renders the SearchBar component if showSearchBar is true', () => {
    const { getByPlaceholderText } = render(<Navbar showSearchBar={true} />);
    expect(getByPlaceholderText('Search')).toBeInTheDocument();  // Assuming SearchBar has a placeholder "Search"
  });

  test('does not render the SearchBar component if showSearchBar is false', () => {
    const { queryByPlaceholderText } = render(<Navbar showSearchBar={false} />);
    expect(queryByPlaceholderText('Search')).not.toBeInTheDocument();
  });

  test('toggles the Menu component when MenuToggleButton is clicked', () => {
    const { getByText, queryByText } = render(<Navbar />);
    const toggleButton = getByText('Toggle Menu');  // Assuming MenuToggleButton renders a text "Toggle Menu"
    fireEvent.click(toggleButton);
    expect(getByText('Menu Item 1')).toBeInTheDocument();  // Assuming Menu renders "Menu Item 1"
    fireEvent.click(toggleButton);
    expect(queryByText('Menu Item 1')).not.toBeInTheDocument();
  });
});
