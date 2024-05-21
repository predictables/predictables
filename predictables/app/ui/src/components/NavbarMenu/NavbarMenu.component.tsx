import './NavbarMenu.styles.css';

const NavbarMenu: React.FC<{ items: string[] }> = ({ items = [] }) => {
  return (
    <div className="menu">
      {items.map((item) => (
        <div className="menu-item">{item}</div>
      ))}
    </div>
  );
};

export default NavbarMenu;
