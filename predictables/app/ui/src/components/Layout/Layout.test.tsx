import { render } from '@testing-library/react';
import Layout from './Layout.component';
import '@testing-library/jest-dom/extend-expect';

declare global {
  namespace jest {
    interface Matchers<R, T> {
      toBeInTheDocument(): R; // Add the toBeInTheDocument function to the type declaration
    }
  }
}


describe('Layout Component', () => {


test('renders the children passed to it', () => {
    const { getByText } = render(
        <Layout>
            <div>Test Content</div>
        </Layout>
    );
    expect(getByText('Test Content')).toBeInTheDocument();
});

  test('renders the Navbar component', () => {
    const { getByText } = render(
      <Layout>
        <div>Test Content</div>
      </Layout>
    );
    expect(getByText('Navbar')).toBeInTheDocument();
  });
});
