import bcDat from '@data/bc.json';

export const fetchPlaceholderData = () => {
  return new Promise((resolve) => {
    // Simulate network delay
    setTimeout(() => {
      // Mock data that the API might return
      const mockApiResponse = {
        data: bcDat['data'],
        message: 'Data fetched successfully',
      };

      resolve(mockApiResponse);
    }, 1500); // Delay in milliseconds
  });
};
