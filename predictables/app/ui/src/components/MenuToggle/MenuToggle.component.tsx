import './MenuToggle.styles.css';

interface MenuToggleProps {
  toggleMenu: () => void;
}

const MenuToggle: React.FC<MenuToggleProps> = ({ toggleMenu }) => {
  return (
    <button className="menu-toggle-button" onClick={toggleMenu}>
      Toggle Menu
    </button>
  );
};

export default MenuToggle;
