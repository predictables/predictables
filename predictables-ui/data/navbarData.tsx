// Navbar data are structured as a list of ItemObj's, where each ItemObj has a name and children.
// Each child has a name and route, and optionally its own children, which also have the same
// Child type.

// interface ItemObj {
//   name: string;
//   children?: Child[];
// }

// interface Child {
//   name: string;
//   route?: string;
//   children?: Child[];
// }

// const edaChild: Child[] = [
//   {
//     name: "Univariate",
//     route: "/eda/univariate",
//   },
//   {
//     name: "Load Data",
//     route: "/eda/univariate/load-data",
//   },
// ];

// const test2ndChild: Child[] = [
//   {
//     name: "Univariate",
//     route: "/eda/univariate",
//   },
//   {
//     name: "Load Data",
//     route: "/eda/univariate/load-data",
//   },
//   {
//     name: "home lol",
//     route: "/",
//   },
// ];

// const test3rdChild: Child[] = [
//   {
//     name: "Univariate",
//     route: "/eda/univariate",
//   },
//   {
//     name: "Load Data",
//     route: "/eda/univariate/load-data",
//   },
// ];

interface NavButton {
  name: string;
  route: string;
}

const data: NavButton[] = [
  {
    name: 'home',
    route: '/',
  },
  {
    name: 'univariate',
    route: '/univariate',
  },
  {
    name: 'multivariate',
    route: '/multivariate',
  },
];

export { data };

export type { NavButton };
