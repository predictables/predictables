import { ReactNode } from 'react';
import Navbar from '../Navbar/Navbar.component';
import './Layout.styles.css';

interface LayoutProps {
  children: ReactNode;
}

const Layout: React.FC<LayoutProps> = ({ children }) => {
  return (
    <div className="layout">
      <Navbar />
      <main className="content">
        {children}
      </main>
    </div>
  );
};

export default Layout;
