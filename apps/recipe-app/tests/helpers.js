/**
 * Test Helpers — HTTP utilities, assertion helpers, and data factories
 *
 * Provides convenience functions for making requests against the Express app,
 * common test data sets, and assertion utilities.
 */

/** Dietary labels — must match the DIETARY_LABELS constant in the app */
const DIETARY_LABELS = [
  'vegetarian',
  'vegan',
  'gluten-free',
  'dairy-free',
  'nut-free',
  'diabetic-friendly',
  'low-sodium',
  'low-carb',
  'keto',
  'paleo',
];

/** SAFE_FOR_MOM preset — diabetic-friendly + low-sodium */
const SAFE_FOR_MOM = ['diabetic-friendly', 'low-sodium'];

/** Days of the week allowed for meal planning */
const VALID_DAYS = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday'];

/** Meal types */
const MEAL_TYPES = ['breakfast', 'lunch', 'dinner', 'snack'];

/** Factory: generate a recipe payload for API/GraphQL calls */
function recipePayload(overrides = {}) {
  return {
    title: overrides.title || 'Factory Recipe',
    ingredients: overrides.ingredients || 'flour, sugar, butter',
    instructions: overrides.instructions || 'Mix ingredients. Bake at 350F for 25 minutes.',
    dietary_tags: overrides.dietary_tags || [],
    image_url: overrides.image_url || '',
    prep_time: overrides.prep_time || 20,
  };
}

/** Factory: generate a user payload for registration */
function userPayload(overrides = {}) {
  return {
    username: overrides.username || `user_${Date.now()}`,
    email: overrides.email || `user_${Date.now()}@test.com`,
    password: overrides.password || 'securepass123',
    display_name: overrides.display_name || 'Test User',
    dietary_preferences: overrides.dietary_preferences || [],
  };
}

/** Factory: generate a meal plan payload */
function mealPlanPayload(overrides = {}) {
  return {
    week_start: overrides.week_start || '2026-02-16',
    name: overrides.name || 'Test Week Plan',
  };
}

/** Factory: generate an ingredient payload */
function ingredientPayload(overrides = {}) {
  return {
    name: overrides.name || 'flour',
    amount: overrides.amount || 2.0,
    unit: overrides.unit || 'cups',
    category: overrides.category || 'dry goods',
  };
}

/** Assert that a recipe object has all expected fields */
function assertRecipeShape(recipe) {
  expect(recipe).toHaveProperty('id');
  expect(recipe).toHaveProperty('title');
  expect(recipe).toHaveProperty('ingredients');
  expect(recipe).toHaveProperty('instructions');
  expect(recipe).toHaveProperty('created_at');
}

/** Assert that a user object has all expected fields (no password_hash) */
function assertUserShape(user) {
  expect(user).toHaveProperty('id');
  expect(user).toHaveProperty('username');
  expect(user).toHaveProperty('email');
  expect(user).toHaveProperty('display_name');
  expect(user).not.toHaveProperty('password_hash');
}

/** Assert an array is sorted by a field in descending order */
function assertSortedDesc(arr, field) {
  for (let i = 1; i < arr.length; i++) {
    expect(arr[i - 1][field] >= arr[i][field]).toBe(true);
  }
}

module.exports = {
  DIETARY_LABELS,
  SAFE_FOR_MOM,
  VALID_DAYS,
  MEAL_TYPES,
  recipePayload,
  userPayload,
  mealPlanPayload,
  ingredientPayload,
  assertRecipeShape,
  assertUserShape,
  assertSortedDesc,
};
