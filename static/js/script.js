const STORAGE_KEYS = {
    USER_ID: "user_id",
    SEARCH_HISTORY: "searchHistory",
    MENU: "menu",
    RATINGS: "recipe_ratings"
};

// Khởi tạo biến toàn cục
let storedRecipes = [];
let selectedRecipe = null;
let suggestedMealPlan = null; // Ensure global declaration
let ratings = JSON.parse(localStorage.getItem(STORAGE_KEYS.RATINGS)) || {};
let DOM_ELEMENTS = {};

// Cấu hình API
const API_CONFIG = {
    BASE_URL: "", // Sử dụng URL tương đối vì backend và frontend cùng domain
    TIMEOUT: 30000,
    MAX_RETRIES: 3,
    RETRY_DELAY: 1000
};

// Cấu hình cache
const CACHE_CONFIG = {
    DB_NAME: "RecipeCacheDB",
    DB_VERSION: 1,
    STORE_NAME: "search_cache",
    MAX_SIZE: 50,
    EXPIRY_TIME: 60 * 60 * 1000 // 1 giờ
};

// Quản lý IndexedDB
function openDatabase() {
    return new Promise((resolve, reject) => {
        const request = indexedDB.open(CACHE_CONFIG.DB_NAME, CACHE_CONFIG.DB_VERSION);
        request.onerror = () => reject("❌ Không thể mở IndexedDB.");
        request.onsuccess = () => resolve(request.result);
        request.onupgradeneeded = (event) => {
            const db = event.target.result;
            if (!db.objectStoreNames.contains(CACHE_CONFIG.STORE_NAME)) {
                db.createObjectStore(CACHE_CONFIG.STORE_NAME);
            }
        };
    });
}

async function getCache(query) {
    const db = await openDatabase();
    return new Promise((resolve) => {
        const tx = db.transaction(CACHE_CONFIG.STORE_NAME, "readonly");
        const store = tx.objectStore(CACHE_CONFIG.STORE_NAME);
        const request = store.get(query.toLowerCase());

        request.onsuccess = () => {
            const result = request.result;
            if (result && Date.now() - result.timestamp < CACHE_CONFIG.EXPIRY_TIME) {
                resolve(result.data);
            } else {
                resolve(null);
            }
        };
        request.onerror = () => resolve(null);
    });
}

async function setCache(query, data) {
    const db = await openDatabase();
    return new Promise((resolve, reject) => {
        const tx = db.transaction(CACHE_CONFIG.STORE_NAME, "readwrite");
        const store = tx.objectStore(CACHE_CONFIG.STORE_NAME);
        store.put({ data, timestamp: Date.now() }, query.toLowerCase());
        tx.oncomplete = () => resolve();
        tx.onerror = () => reject("❌ Lỗi khi lưu cache.");
    });
}

async function trimOldCache(maxItems = CACHE_CONFIG.MAX_SIZE) {
    try {
        const db = await openDatabase();
        const tx = db.transaction(CACHE_CONFIG.STORE_NAME, "readwrite");
        const store = tx.objectStore(CACHE_CONFIG.STORE_NAME);

        const items = [];
        store.openCursor().onsuccess = function (event) {
            const cursor = event.target.result;
            if (cursor) {
                items.push({ query: cursor.key, timestamp: cursor.value.timestamp });
                cursor.continue();
            } else {
                items.sort((a, b) => b.timestamp - a.timestamp);
                const itemsToDelete = items.slice(maxItems);
                for (const item of itemsToDelete) {
                    store.delete(item.query);
                }
            }
        };
        tx.oncomplete = () => console.log(`🧹 Đã dọn cache, giữ lại ${Math.min(items.length, maxItems)} truy vấn gần nhất.`);
    } catch (err) {
        console.error("❌ trimOldCache thất bại:", err);
    }
}

// Tính điểm DQI-I dựa trên bảng chuẩn
function calculateDQII(nutrients, ingredients) {
    let score = { variety_food: 0, variety_protein: 0, adequacy: 0, moderation: 0, balance: 0, total: 0 };

    const foodGroups = new Set(ingredients.map(ing => {
        const ingLower = ing.toLowerCase();
        if (ingLower.includes("chickpea") || ingLower.includes("bean")) return "dairy/beans";
        if (ingLower.includes("onion") || ingLower.includes("garlic") || ingLower.includes("parsley") || ingLower.includes("salad")) return "vegetables";
        if (ingLower.includes("flour") || ingLower.includes("bread") || ingLower.includes("pita")) return "grains";
        if (ingLower.includes("cream") || ingLower.includes("custard") || ingLower.includes("banana")) return "other";
        if (ingLower.includes("oil")) return "other";
        return "other";
    }));
    score.variety_food = foodGroups.size >= 1 ? Math.min(foodGroups.size * 3, 15) : 0;

    const proteinSources = new Set(ingredients.map(ing => {
        const ingLower = ing.toLowerCase();
        if (ingLower.includes("cream") || ingLower.includes("custard")) return "dairy";
        if (ingLower.includes("chickpea") || ingLower.includes("bean")) return "beans";
        return null;
    }).filter(Boolean));
    score.variety_protein = proteinSources.size >= 3 ? 5 : proteinSources.size === 2 ? 3 : proteinSources.size === 1 ? 1 : 0;

    const rda = { protein: 50, fibre: 25, kcal: 2000 };
    score.adequacy = 0;
    if (nutrients.protein) score.adequacy += Math.min((nutrients.protein / rda.protein) * 5, 5);
    if (nutrients.fibre) score.adequacy += Math.min((nutrients.fibre / rda.fibre) * 5, 5);
    if (nutrients.kcal) score.adequacy += Math.min((nutrients.kcal / rda.kcal) * 20, 20);
    score.adequacy = Math.min(score.adequacy, 40);

    const maxLimits = { fat: 65, saturates: 20, salt: 2.3 };
    score.moderation = 0;
    if (nutrients.fat) score.moderation += Math.min(((maxLimits.fat - nutrients.fat) / maxLimits.fat) * 6, 6);
    if (nutrients.saturates) score.moderation += Math.min(((maxLimits.saturates - nutrients.saturates) / maxLimits.saturates) * 6, 6);
    if (nutrients.salt) score.moderation += Math.min(((maxLimits.salt - nutrients.salt) / maxLimits.salt) * 6, 6);
    score.moderation = Math.min(score.moderation, 30);

    if (nutrients.kcal) {
        const proteinPerc = (nutrients.protein * 4 / nutrients.kcal) * 100;
        const fatPerc = (nutrients.fat * 9 / nutrients.kcal) * 100;
        const carbsPerc = (nutrients.carbs * 4 / nutrients.kcal) * 100;
        const idealRange = { protein: [10, 35], fat: [20, 35], carbs: [45, 65] };
        score.balance = (
            (idealRange.protein[0] <= proteinPerc && proteinPerc <= idealRange.protein[1] ? 5 : 0) +
            (idealRange.fat[0] <= fatPerc && fatPerc <= idealRange.fat[1] ? 5 : 0)
        ) / 2;
    }

    score.total = score.variety_food + score.variety_protein + score.adequacy + score.moderation + score.balance;
    return score;
}

// Chuẩn hóa dữ liệu dinh dưỡng
function normalizeNutrients(nutrients) {
    function parseNutrient(value) {
        if (!value || value === "0") return 0.0;
        const cleanedValue = String(value).replace(/[^\d.]/g, '');
        return parseFloat(cleanedValue) || 0.0;
    }

    return {
        kcal: parseNutrient(nutrients.kcal),
        fat: parseNutrient(nutrients.fat),
        saturates: parseNutrient(nutrients.saturates),
        carbs: parseNutrient(nutrients.carbs),
        sugars: parseNutrient(nutrients.sugars),
        fibre: parseNutrient(nutrients.fibre),
        protein: parseNutrient(nutrients.protein),
        salt: parseNutrient(nutrients.salt)
    };
}

// Hàm gửi yêu cầu với retry
async function fetchWithRetry(url, options, retries = API_CONFIG.MAX_RETRIES) {
    for (let i = 0; i < retries; i++) {
        try {
            const response = await fetch(url, {
                ...options,
                signal: AbortSignal.timeout(API_CONFIG.TIMEOUT)
            });
            if (!response.ok) throw new Error(`HTTP error: ${response.status}`);
            return await response.json();
        } catch (error) {
            if (i === retries - 1) throw error;
            console.warn(`Retrying (${i + 1}/${retries})...`);
            await new Promise(resolve => setTimeout(resolve, API_CONFIG.RETRY_DELAY));
        }
    }
}

// 🔍 Tìm kiếm món ăn
async function searchRecipes() {
    const query = DOM_ELEMENTS.userInput ? DOM_ELEMENTS.userInput.value.trim() : "";
    const userId = localStorage.getItem(STORAGE_KEYS.USER_ID) || generateUserId();
    const diet = DOM_ELEMENTS.dietSelect ? DOM_ELEMENTS.dietSelect.value : "all";
    const cacheKey = `${query || "diet"}_${diet}`;

    if (!query && diet === "all") {
        if (DOM_ELEMENTS.recipeList) {
            DOM_ELEMENTS.recipeList.innerHTML = "";
        }
        if (DOM_ELEMENTS.bestRecipeContainer) {
            DOM_ELEMENTS.bestRecipeContainer.style.display = "none";
        }
        if (DOM_ELEMENTS.suggestedMealPlan) {
            DOM_ELEMENTS.suggestedMealPlan.innerHTML = `<p class="text-muted">Nhấn tìm kiếm để nhận gợi ý thực đơn.</p>`;
        }
        suggestedMealPlan = null; // Reset suggestedMealPlan
        updateAddMealPlanButtonState();
        return;
    }

    const validDiets = ["all", "vegetarian", "keto", "gluten_free"];
    if (!validDiets.includes(diet)) {
        console.error(`Invalid diet value: ${diet}`);
        showError("Invalid diet");
        return;
    }

    console.log("Sending request to /search:", { ingredients: query, user_id: userId, diet: diet });

    const cachedData = await getCache(cacheKey);
    if (cachedData) {
        console.log("Data retrieved from IndexedDB cache!");
        renderRecipeList(cachedData);
        return;
    }

    showLoading(true);
    try {
        const response = await fetchWithRetry("/search", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ ingredients: query, user_id: userId, diet: diet })
        });

        console.log("Response data:", response);
        showLoading(false);

        if (response.error) {
            console.error("Server returned error:", response.error);
            return showError(response.error);
        }
        if (!Array.isArray(response) || response.length === 0) return showError("No recipes found");

        await setCache(cacheKey, response);
        trimOldCache();
        renderRecipeList(response);
    } catch (error) {
        console.error("Fetch API error:", error);
        showError(error.message || "Error loading data from server");
    } finally {
        showLoading(false);
    }
}

// 📋 Gợi ý thực đơn
async function suggestMealPlan() {
    const query = DOM_ELEMENTS.userInput ? DOM_ELEMENTS.userInput.value.trim() : "";
    const userId = localStorage.getItem(STORAGE_KEYS.USER_ID) || generateUserId();
    const diet = DOM_ELEMENTS.dietSelect ? DOM_ELEMENTS.dietSelect.value : "all";
    const cacheKey = `meal_plan_${query || "diet"}_${diet}`;

    if (!query && diet === "all") {
        if (DOM_ELEMENTS.suggestedMealPlan) {
            DOM_ELEMENTS.suggestedMealPlan.innerHTML = `<p class="text-muted">Nhấn tìm kiếm để nhận gợi ý thực đơn.</p>`;
        }
        suggestedMealPlan = null; // Reset suggestedMealPlan
        updateAddMealPlanButtonState();
        return;
    }

    const validDiets = ["all", "vegetarian", "keto", "gluten_free"];
    if (!validDiets.includes(diet)) {
        console.error(`Invalid diet value: ${diet}`);
        showError("Invalid diet");
        return;
    }

    console.log("Sending request to /meal_plan:", { ingredients: query, user_id: userId, diet: diet });

    const cachedData = await getCache(cacheKey);
    if (cachedData) {
        console.log("Meal plan retrieved from IndexedDB cache!");
        renderMealPlan(cachedData);
        return;
    }

    showLoading(true);
    try {
        const response = await fetchWithRetry("/meal_plan", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ ingredients: query, user_id: userId, diet: diet })
        });

        console.log("Meal plan data:", response);
        showLoading(false);

        if (response.error) {
            console.error("Server returned error:", response.error);
            return showError(response.error);
        }

        await setCache(cacheKey, response);
        trimOldCache();
        renderMealPlan(response);
    } catch (error) {
        console.error("Fetch API error:", error);
        showError(error.message || "Error loading meal plan from server");
    } finally {
        showLoading(false);
    }
}

function renderMealPlan(data) {
    suggestedMealPlan = data; // Set global variable
    const mealPlanDiv = DOM_ELEMENTS.suggestedMealPlan;

    if (mealPlanDiv) {
        if (!data || (!data.appetizer && !data.main && !data.dessert)) {
            mealPlanDiv.innerHTML = `<p class="text-danger">Không tìm thấy thực đơn phù hợp.</p>`;
            suggestedMealPlan = null; // Reset if no valid meal plan
            updateAddMealPlanButtonState();
            return;
        }

        const categories = [
            { key: 'appetizer', label: 'Món khai vị' },
            { key: 'main', label: 'Món chính' },
            { key: 'dessert', label: 'Món tráng miệng' }
        ];

        let html = '<h5>Thực đơn gợi ý:</h5><ul class="list-group">';
        categories.forEach(category => {
            const recipe = data[category.key];
            if (recipe) {
                const safeRecipe = encodeURIComponent(JSON.stringify(escapeSpecialChars(recipe)));
                const avgRating = recipe.avg_rating ? `⭐ ${recipe.avg_rating.toFixed(1)}/5` : "Chưa có đánh giá";
                html += `
                    <li class="list-group-item" data-recipe-id="${recipe.id || 'no-id'}" data-recipe="${safeRecipe}" onclick="showRecipeDetail(this)">
                        <strong>${category.label}:</strong> ${recipe.name || "Không có tên"}
                        <span class="float-end text-muted">${avgRating}</span>
                    </li>
                `;
            }
        });
        html += '</ul>';

        // Hiển thị tóm tắt dinh dưỡng
        if (data.nutrition_summary) {
            const { calories, protein, fat, carbs } = data.nutrition_summary;
            html += `
                <h5 class="mt-3">Tóm tắt dinh dưỡng:</h5>
                <p>Calo: ${calories.toFixed(1)} kcal | Protein: ${protein.toFixed(1)}g | Chất béo: ${fat.toFixed(1)}g | Carb: ${carbs.toFixed(1)}g</p>
            `;
        }

        mealPlanDiv.innerHTML = html;
        updateAddMealPlanButtonState();
    }
}

function updateAddMealPlanButtonState() {
    const button = document.querySelector('button.btn-success[onclick="addSuggestedMealPlan()"]');
    if (button) {
        button.disabled = !suggestedMealPlan || (!suggestedMealPlan.appetizer && !suggestedMealPlan.main && !suggestedMealPlan.dessert);
    }
}

function addSuggestedMealPlan() {
    if (!suggestedMealPlan || (!suggestedMealPlan.appetizer && !suggestedMealPlan.main && !suggestedMealPlan.dessert)) {
        showModal("⚠ Không có thực đơn gợi ý để thêm!", "warning");
        return;
    }

    let menu = JSON.parse(localStorage.getItem(STORAGE_KEYS.MENU)) || [];
    let added = false;

    const categories = ['appetizer', 'main', 'dessert'];
    categories.forEach(category => {
        const recipe = suggestedMealPlan[category];
        if (recipe && !menu.some(item => item.id === recipe.id)) {
            menu.push(recipe);
            added = true;
        }
    });

    if (added) {
        localStorage.setItem(STORAGE_KEYS.MENU, JSON.stringify(menu));
        showModal("✅ Đã thêm thực đơn gợi ý vào thực đơn của bạn!", "success");
        updateMealPlanUI();
    } else {
        showModal("⚠ Tất cả món trong thực đơn gợi ý đã có trong thực đơn của bạn!", "warning");
    }
}

function showLoading(isShow) {
    if (DOM_ELEMENTS.loadingOverlay) {
        DOM_ELEMENTS.loadingOverlay.style.display = isShow ? "flex" : "none";
    }
}

function showError(message) {
    if (DOM_ELEMENTS.recipeList) {
        DOM_ELEMENTS.recipeList.innerHTML = `<li class='list-group-item text-danger'>❌ ${message}</li>`;
    }
    if (DOM_ELEMENTS.bestRecipeContainer) {
        DOM_ELEMENTS.bestRecipeContainer.style.display = "none";
    }
    if (DOM_ELEMENTS.suggestedMealPlan) {
        DOM_ELEMENTS.suggestedMealPlan.innerHTML = `<p class="text-danger">❌ ${message}</p>`;
        suggestedMealPlan = null; // Reset suggestedMealPlan on error
        updateAddMealPlanButtonState();
    }
    showModal(message, "error");
}

function renderRecipeList(data) {
    storedRecipes = data;
    const bestRecipeContainer = DOM_ELEMENTS.bestRecipeContainer;
    const bestRecipeDiv = DOM_ELEMENTS.bestRecipe;
    const recipeList = DOM_ELEMENTS.recipeList;

    if (bestRecipeContainer && bestRecipeDiv && recipeList) {
        if (data.length === 0) {
            bestRecipeContainer.style.display = "none";
            recipeList.innerHTML = `<li class="list-group-item text-muted">Không tìm thấy món ăn nào.</li>`;
            suggestedMealPlan = null; // Reset suggestedMealPlan
            updateAddMealPlanButtonState();
            return;
        }

        if (data.length > 1 && data[0].name) {
            const bestRecipe = data[0];
            const safeBestRecipe = encodeURIComponent(JSON.stringify(escapeSpecialChars(bestRecipe)));
            const bestAvgRating = bestRecipe.avg_rating ? `⭐ ${bestRecipe.avg_rating.toFixed(1)}/5` : "Chưa có đánh giá";
            bestRecipeDiv.innerHTML = `
                <div class="list-group-item" data-recipe-id="${bestRecipe.id || 'no-id'}" data-recipe="${safeBestRecipe}" onclick="showRecipeDetail(this)">
                    <h5>🍽️ ${bestRecipe.name || "Không có tên"} <span class="float-end text-muted">${bestAvgRating}</span></h5>
                    <p class="mb-0">${(bestRecipe.ingredients || []).slice(0, 3).join(", ")}${bestRecipe.ingredients?.length > 3 ? "..." : ""}</p>
                </div>
            `;
            bestRecipeContainer.style.display = "block";

            recipeList.innerHTML = data.slice(1).map(recipe => {
                const recipeId = recipe.id || "no-id";
                const avgRating = recipe.avg_rating ? `⭐ ${recipe.avg_rating.toFixed(1)}/5` : "Chưa có đánh giá";
                const safeRecipe = encodeURIComponent(JSON.stringify(escapeSpecialChars(recipe)));
                return `
                    <li class="list-group-item" data-recipe-id="${recipeId}" data-recipe="${safeRecipe}" onclick="showRecipeDetail(this)">
                        🍽️ ${recipe.name || "Không có tên"}
                        <span class="float-end text-muted">${avgRating}</span>
                    </li>
                `;
            }).join("");
        } else {
            bestRecipeContainer.style.display = "none";
            recipeList.innerHTML = data.map(recipe => {
                const recipeId = recipe.id || "no-id";
                const avgRating = recipe.avg_rating ? `⭐ ${recipe.avg_rating.toFixed(1)}/5` : "Chưa có đánh giá";
                const safeRecipe = encodeURIComponent(JSON.stringify(escapeSpecialChars(recipe)));
                return `
                    <li class="list-group-item" data-recipe-id="${recipeId}" data-recipe="${safeRecipe}" onclick="showRecipeDetail(this)">
                        🍽️ ${recipe.name || "Không có tên"}
                        <span class="float-end text-muted">${avgRating}</span>
                    </li>
                `;
            }).join("");
        }
    }

    saveSearchHistory(DOM_ELEMENTS.userInput?.value.trim());
    updateSearchHistoryUI();
}

// escape/unescape
function escapeSpecialChars(obj) {
    if (typeof obj !== "object" || obj === null) return obj;
    if (Array.isArray(obj)) return obj.map(escapeSpecialChars);
    return Object.fromEntries(
        Object.entries(obj).map(([key, value]) =>
            [key, typeof value === "string" ? value.replace(/"/g, '\\"').replace(/\n/g, '\\n').replace(/\r/g, '\\r').replace(/\\t/g, '\\t') : escapeSpecialChars(value)]
        )
    );
}

function unescapeSpecialChars(obj) {
    if (typeof obj !== "object" || obj === null) return obj;
    if (Array.isArray(obj)) return obj.map(unescapeSpecialChars);
    return Object.fromEntries(
        Object.entries(obj).map(([key, value]) =>
            [key, typeof value === "string" ? value.replace(/\\"/g, '"').replace(/\\n/g, '\n').replace(/\\r/g, '\r').replace(/\\t/g, '\t') : unescapeSpecialChars(value)]
        )
    );
}

// 📋 Chi tiết món ăn
function showRecipeDetail(element) {
    try {
        const recipeStr = element.getAttribute("data-recipe");
        if (!recipeStr) throw new Error("Không tìm thấy dữ liệu món ăn");

        const decodedRecipe = decodeURIComponent(recipeStr);
        const recipe = unescapeSpecialChars(JSON.parse(decodedRecipe));
        selectedRecipe = recipe;

        const nutrients = normalizeNutrients(recipe.nutrients || {});
        const dqiScore = calculateDQII(nutrients, recipe.ingredients || []);
        const steps = recipe.steps || [];
        const stepsHtml = steps.map((step, index) => `<li><strong>Bước ${index + 1}:</strong> ${step.trim()}</li>`).join("");
        const avgRating = recipe.avg_rating ? `⭐ Trung bình: ${recipe.avg_rating.toFixed(1)}/5` : "Chưa có đánh giá";
        const nutritionStatus = nutrients.kcal === 0 ? "<p class='text-warning'>Không có dữ liệu dinh dưỡng để đánh giá</p>" : "";

        if (DOM_ELEMENTS.recipeDetail) {
            DOM_ELEMENTS.recipeDetail.innerHTML = `
                <h4>${recipe.name || "Không có tên"}</h4>
                ${recipe.image ? `<img src="${recipe.image}" alt="${recipe.name || "Hình ảnh"}" class="img-thumbnail mb-3" style="max-width: 200px; display: block; margin: 0 auto;">` : ""}
                <p><strong>Nguyên liệu:</strong> ${(recipe.ingredients || []).join(", ") || "Không có thông tin"}</p>
                <h5>📝 Cách làm:</h5>
                <ul>${stepsHtml}</ul>
                <h5>🍎 Đánh giá dinh dưỡng (DQI-I):</h5>
                ${nutritionStatus}
                <p>Đa dạng (Food Groups): ${dqiScore.variety_food.toFixed(2)}/15 | Đa dạng (Protein): ${dqiScore.variety_protein.toFixed(2)}/5 | Đủ chất: ${dqiScore.adequacy.toFixed(2)}/40 | Điều độ: ${dqiScore.moderation.toFixed(2)}/30 | Cân bằng: ${dqiScore.balance.toFixed(2)}/10</p>
                <p><strong>Tổng điểm DQI-I: ${dqiScore.total.toFixed(2)}/100</strong></p>
                <h5>⭐ Đánh giá người dùng:</h5>
                <p>${avgRating}</p>
                <select id="rating-select" class="form-control" onchange="rateRecipe('${recipe.id || "no-id"}', this.value)">
                    <option value="">Chọn đánh giá</option>
                    <option value="1">⭐ 1 Sao</option>
                    <option value="2">⭐⭐ 2 Sao</option>
                    <option value="3">⭐⭐⭐ 3 Sao</option>
                    <option value="4">⭐⭐⭐⭐ 4 Sao</option>
                    <option value="5">⭐⭐⭐⭐⭐ 5 Sao</option>
                </select>
                <button class="btn btn-warning mt-2" onclick="addToMenu()">📋 Lên thực đơn</button>
            `;
        }
    } catch (error) {
        console.error("Lỗi khi phân tích dữ liệu món ăn:", error);
        if (DOM_ELEMENTS.recipeDetail) {
            DOM_ELEMENTS.recipeDetail.innerHTML = "<p class='text-danger'>Lỗi khi tải chi tiết món ăn.</p>";
        }
        showModal("Lỗi khi tải chi tiết món ăn: " + error.message, "error");
    }
}

// ⭐ Đánh giá
async function rateRecipe(recipeId, rating) {
    if (!rating) return;

    const userId = localStorage.getItem(STORAGE_KEYS.USER_ID) || generateUserId();
    try {
        const response = await fetchWithRetry("/feedback", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ user_id: userId, recipe_name: storedRecipes.find(r => r.id === recipeId)?.name || "Unknown", rating: parseInt(rating) })
        });

        console.log("Feedback response:", response);
        if (response.error) {
            console.error("Server returned error:", response.error);
            showModal("Lỗi khi gửi đánh giá: " + response.error, "error");
            return;
        }

        ratings[recipeId] = ratings[recipeId] || [];
        ratings[recipeId].push(parseInt(rating));
        localStorage.setItem(STORAGE_KEYS.RATINGS, JSON.stringify(ratings));

        const avgRating = ratings[recipeId].reduce((a, b) => a + b, 0) / ratings[recipeId].length;
        updateRatingUI(recipeId, avgRating);

        showModal(`Bạn đã đánh giá món ăn ${rating} sao!`, "success");
        console.log(`⭐ Đánh giá: ${recipeId} = ${rating}`);
    } catch (error) {
        console.error("Feedback error:", error);
        showModal("Lỗi khi gửi đánh giá: " + error.message, "error");
    }
}

function updateRatingUI(recipeId, avgRating) {
    const recipeItems = DOM_ELEMENTS.recipeList?.querySelectorAll(`li[data-recipe-id="${recipeId}"]`);
    if (recipeItems) {
        recipeItems.forEach(item => item.querySelector(".float-end").textContent = `⭐ ${avgRating.toFixed(1)}/5`);
    }

    const bestRecipeItem = DOM_ELEMENTS.bestRecipe?.querySelector(`div[data-recipe-id="${recipeId}"]`);
    if (bestRecipeItem) {
        bestRecipeItem.querySelector(".float-end").textContent = `⭐ ${avgRating.toFixed(1)}/5`);
    }

    const mealPlanItems = DOM_ELEMENTS.suggestedMealPlan?.querySelectorAll(`li[data-recipe-id="${recipeId}"]`);
    if (mealPlanItems) {
        mealPlanItems.forEach(item => item.querySelector(".float-end").textContent = `⭐ ${avgRating.toFixed(1)}/5`);
    }

    if (selectedRecipe && selectedRecipe.id === recipeId && DOM_ELEMENTS.recipeDetail) {
        const detailRating = DOM_ELEMENTS.recipeDetail.querySelector("p:nth-child(4)");
        if (detailRating) detailRating.textContent = `⭐ Trung bình: ${avgRating.toFixed(1)}/5`;

        const recipeIndex = storedRecipes.findIndex(r => r.id === recipeId);
        if (recipeIndex !== -1) storedRecipes[recipeIndex].avg_rating = avgRating;
    }
}

// 📋 Thực đơn
function addToMenu() {
    if (!selectedRecipe) {
        showModal("⚠ Không có món nào được chọn!", "warning");
        return;
    }

    let menu = JSON.parse(localStorage.getItem(STORAGE_KEYS.MENU)) || [];
    if (!menu.some(item => item.id === selectedRecipe.id)) {
        menu.push(selectedRecipe);
        localStorage.setItem(STORAGE_KEYS.MENU, JSON.stringify(menu));
        showModal(`✅ Đã thêm "${selectedRecipe.name}" vào thực đơn!`, "success");
    } else {
        showModal(`⚠ "${selectedRecipe.name}" đã có trong thực đơn!`, "warning");
    }

    updateMealPlanUI();
}

function updateMealPlanUI() {
    if (DOM_ELEMENTS.mealPlanList) {
        const mealPlan = JSON.parse(localStorage.getItem(STORAGE_KEYS.MENU)) || [];
        DOM_ELEMENTS.mealPlanList.innerHTML = mealPlan.length === 0
            ? "<li class='list-group-item text-muted'>Chưa có món nào trong thực đơn</li>"
            : mealPlan.map(recipe => `
                <li class='list-group-item d-flex justify-content-between align-items-center'>
                    🍽️ ${recipe.name}
                    <button class='btn btn-sm btn-danger' onclick="removeFromMealPlan('${recipe.id}')">❌</button>
                </li>
            `).join("");
    }
}

function removeFromMealPlan(recipeId) {
    let mealPlan = JSON.parse(localStorage.getItem(STORAGE_KEYS.MENU)) || [];
    mealPlan = mealPlan.filter(recipe => recipe.id !== recipeId);
    localStorage.setItem(STORAGE_KEYS.MENU, JSON.stringify(mealPlan));
    showModal("❌ Đã xóa món khỏi thực đơn!", "warning");
    updateMealPlanUI();
}

function clearMealPlan() {
    localStorage.removeItem(STORAGE_KEYS.MENU);
    showModal("❌ Đã xóa tất cả món trong thực đơn!", "warning");
    updateMealPlanUI();
}

// 📜 Lịch sử tìm kiếm
function saveSearchHistory(query) {
    let history = JSON.parse(localStorage.getItem(STORAGE_KEYS.SEARCH_HISTORY)) || [];
    if (!history.includes(query) && query) {
        history.unshift(query);
        if (history.length > 5) history.pop();
        localStorage.setItem(STORAGE_KEYS.SEARCH_HISTORY, JSON.stringify(history));
    }
}

function updateSearchHistoryUI() {
    if (DOM_ELEMENTS.searchHistory) {
        const history = JSON.parse(localStorage.getItem(STORAGE_KEYS.SEARCH_HISTORY)) || [];
        DOM_ELEMENTS.searchHistory.innerHTML = history.length === 0
            ? "<li class='list-group-item'>Chưa có tìm kiếm nào</li>"
            : history.map(item => `<li class='list-group-item' onclick="searchFromHistory('${item}')">🔍 ${item}</li>`).join("");
    }
}

function searchFromHistory(query) {
    if (DOM_ELEMENTS.userInput) {
        DOM_ELEMENTS.userInput.value = query;
        searchRecipes();
        suggestMealPlan();
    }
}

function generateUserId() {
    const newUserId = "user_" + Math.random().toString(36).substr(2, 9);
    localStorage.setItem(STORAGE_KEYS.USER_ID, newUserId);
    return newUserId;
}

// Hiển thị modal
function showModal(message, type = "info") {
    const modal = new bootstrap.Modal(document.getElementById('notificationModal'));
    const modalBody = document.getElementById('modalBody');

    modalBody.innerHTML = message;
    let modalClass = 'text-info';
    if (type === 'success') modalClass = 'text-success';
    else if (type === 'error') modalClass = 'text-danger';
    else if (type === 'warning') modalClass = 'text-warning';

    modalBody.className = `modal-body ${modalClass}`;
    modal.show();
}

// 🚀 Khởi động
window.onload = () => {
    DOM_ELEMENTS = {
        userInput: document.getElementById("user-input"),
        loadingOverlay: document.getElementById("loading-overlay"),
        recipeList: document.getElementById("recipe-list"),
        bestRecipe: document.getElementById("best-recipe"),
        bestRecipeContainer: document.getElementById("best-recipe-container"),
        recipeDetail: document.getElementById("recipe-detail"),
        searchHistory: document.getElementById("search-history"),
        mealPlanList: document.getElementById("meal-plan-list"),
        suggestedMealPlan: document.getElementById("suggested-meal-plan"),
        dietSelect: document.getElementById("diet-select")
    };

    if (!DOM_ELEMENTS.dietSelect) {
        console.error("❌ Phần tử dietSelect không tồn tại!");
    } else {
        updateSearchHistoryUI();
        updateMealPlanUI();
        updateAddMealPlanButtonState(); // Initialize button state
    }
};