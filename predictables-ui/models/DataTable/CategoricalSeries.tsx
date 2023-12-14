import DataSeries from './DataSeries';

/**
 * @class CategoricalSeries
 * @classdesc A class for categorical data series. Extends DataSeries to add categorical-specific functionality. Encodes the data as a map of categories to values.
 * @augments DataSeries
 * @param {object} args - The arguments object. See {@link DataSeries} for details.
 * @param {array} additionalCategories - An array of additional categories to add to the series. If you know that your data contains all possible categories, you can leave this blank.
 * @method addCategory - Adds a category to the series.
 * @method removeCategory - Removes a category from the series.
 * @method getCategories - Returns a map of the categories in the series.
 */
class CategoricalSeries extends DataSeries {
  constructor(args: any, additionalCategories?: any[]) {
    super(args);
  }
}
