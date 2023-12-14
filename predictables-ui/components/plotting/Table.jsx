// class Table {
//   constructor(
//     elementID = 'root',
//     data = [],
//     columnNames = [],
//     rowNames = [],
//     headerProps = {
//       fontAlign: 'center',
//       fontFamily: 'InterTight',
//       fontSize: 13,
//       fontColor: 'white',
//       fontBold: true,
//       fontItalic: false,

//       lineWidth: 1,
//       lineColor: 'black',

//       fillColor: 'black',
//     },
//     tableProps = {
//       fontAlign: 'center',
//       fontFamily: 'InterTight',
//       fontSize: 12,
//       fontColor: 'black',
//       fontBold: false,
//       fontItalic: false,

//       lineWidth: 1,
//       lineColor: 'black',
//     },
//   ) {
//     this.elementID = elementID;
//     this.data = data;
//     this.columnNames = columnNames;
//     this.rowNames = rowNames;
//     this.headerProps = headerProps;
//     this.tableProps = tableProps;
//   }

//   setHeaderProps(props) {
//     Object.keys(props).forEach((key) => {
//       if (key in this.headerProps) {
//         this.headerProps[key] = props[key];
//       }
//     });
//   }

//   setTableProps(props) {
//     Object.keys(props).forEach((key) => {
//       if (key in this.tableProps) {
//         this.tableProps[key] = props[key];
//       }
//     });
//   }

//   draw() {
//     const { data, columnNames, rowNames, headerProps, tableProps } = this;

//     // Values are the row names followed by the data one column at a time
//     let values = [rowNames];
//     data.forEach((row) => {
//       values.push(row);
//     });

//     // Header is the column names, wrapped in a <b> tag if headerProps.fontBold is true
//     // and then an <i> tag if headerProps.fontItalic is true
//     const headerValues = columnNames.map((name) => {
//       if (headerProps.fontBold) {
//         name = `<b>${name}</b>`;
//       }
//       if (headerProps.fontItalic) {
//         name = `<i>${name}</i>`;
//       }
//       return name;
//     });

//     // Create the header object
//     const header = {
//       values: headerValues,
//       align: headerProps.fontAlign,
//       line: { width: headerProps.lineWidth, color: headerProps.lineColor },
//       fill: { color: headerProps.fillColor },
//       font: {
//         family: headerProps.fontFamily,
//         size: headerProps.fontSize,
//         color: headerProps.fontColor,
//       },
//     };

//     // Create the cells object
//     const cells = {
//       values: values,
//       align: tableProps.fontAlign,
//       line: { width: tableProps.lineWidth, color: tableProps.lineColor },
//       font: {
//         family: tableProps.fontFamily,
//         size: tableProps.fontSize,
//         color: tableProps.fontColor,
//       },
//     };

//     // Create the data object
//     const dataObj = [
//       {
//         type: 'table',
//         header: header,
//         cells: cells,
//       },
//     ];

//     // Draw the table

//   }
// }

// export default Table;
