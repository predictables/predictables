import { useState } from 'react';
import Logo from '../Logo/Logo.component';
import NavbarSearchbar from '../NavbarSearchbar/NavbarSearchbar.component';
import MenuToggle from '../MenuToggle/MenuToggle.component';
import NavbarMenu from '../NavbarMenu/NavbarMenu.component';
import './Navbar.styles.css';

interface NavbarProps {
  showSearchBar?: boolean;
}

const Navbar: React.FC<NavbarProps> = ({ showSearchBar = true }) => {
  const [isMenuOpen, setIsMenuOpen] = useState(false);

  const toggleMenu = () => {
    setIsMenuOpen(!isMenuOpen);
  };

  return (
    <nav className="navbar">
      <Logo />
      {showSearchBar && <NavbarSearchbar />}
      <MenuToggle toggleMenu={toggleMenu} />
      {isMenuOpen && <NavbarMenu />}
    </nav>
  );
};

export default Navbar;
