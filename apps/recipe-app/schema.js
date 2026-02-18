/**
 * GraphQL Schema Definition (SDL)
 *
 * Introduced: session 12 (GraphQL pivot)
 * Reviewed: session 16 (bug bash — no schema-level issues found)
 *
 * Types mirror the SQLite tables: Recipe, Ingredient, MealPlan,
 * MealPlanItem, GroceryItem, ShareLink, and User (session 18).
 */

const typeDefs = `#graphql
  type Recipe {
    id: Int!
    title: String!
    ingredients: String!
    instructions: String!
    dietaryTags: [String!]!
    imageUrl: String
    prepTime: Int
    createdAt: String
    ingredientList: [Ingredient!]!
    owner: User
  }

  type Ingredient {
    id: Int!
    name: String!
    amount: Float!
    unit: String!
    category: String
  }

  type MealPlan {
    id: Int!
    name: String!
    weekStart: String!
    createdAt: String
    items: [MealPlanItem!]!
    groceryList: [GroceryItem!]!
  }

  type MealPlanItem {
    id: Int!
    dayOfWeek: String!
    mealType: String!
    recipe: Recipe!
  }

  type GroceryItem {
    name: String!
    totalAmount: Float!
    unit: String!
    category: String
    recipeCount: Int!
    fromRecipes: String
  }

  type ShareLink {
    id: Int!
    recipeId: Int!
    code: String!
    createdAt: String
  }

  type User {
    id: Int!
    username: String!
    email: String!
    displayName: String
    dietaryPreferences: [String!]!
    createdAt: String
  }

  type AuthPayload {
    token: String!
    user: User!
  }

  type Query {
    recipes(dietFilter: [String!], safeForMom: Boolean, maxPrepTime: Int): [Recipe!]!
    recipe(id: Int!): Recipe
    searchRecipes(query: String!): [Recipe!]!
    recipeByShareCode(code: String!): Recipe
    dietaryLabels: [String!]!
    mealPlans: [MealPlan!]!
    mealPlan(id: Int!): MealPlan
  }

  type Mutation {
    createRecipe(
      title: String!
      ingredients: String!
      instructions: String!
      dietary_tags: [String!]
      image_url: String
      prep_time: Int
      owner_id: Int
    ): Recipe!

    updateRecipe(
      id: Int!
      title: String
      ingredients: String
      instructions: String
      dietary_tags: [String!]
      image_url: String
      prep_time: Int
    ): Recipe

    deleteRecipe(id: Int!): Boolean!

    shareRecipe(recipeId: Int!): ShareLink

    addIngredients(
      recipeId: Int!
      ingredients: [IngredientInput!]!
    ): [Ingredient!]

    createMealPlan(weekStart: String!, name: String!): MealPlan!

    addMealPlanItem(
      planId: Int!
      recipeId: Int!
      dayOfWeek: String!
      mealType: String!
    ): MealPlanItem

    removeMealPlanItem(id: Int!): Boolean!
  }

  input IngredientInput {
    name: String!
    amount: Float!
    unit: String!
    category: String
  }
`;

module.exports = { typeDefs };
