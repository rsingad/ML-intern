# Beginner's Guide: Restaurant Data Analysis Project

Swagat hai! Agar aap Machine Learning mein naye hain, toh ye guide aapko samjhayegi ki humne is project mein kya kiya aur kaise kiya.

---

## 🚀 1. Project Kya Hai?
Is project mein humne ek restaurant dataset ka use kiya hai. Humne computer ko ye sikhaya ki:
1. Restaurant ki **Rating** predict karna (Regression).
2. Restaurant ki **Cuisine** classify karna (Classification).
3. Naye restaurants **Recommend** karna (Similarity Analysis).
4. **Locations** ka analysis karna (Geographical Analysis).

---

## 🛠 2. Setup (Tiyaari)
Sabse pehle humne ek **Virtual Environment (`venv`)** banaya. 
- **Kyun?** Taaki hum is project ki libraries (jaise `pandas`, `scikit-learn`) ko system ki doosri cheezon se alag safely install kar sakein.

---

## 📊 3. Task 1: Rating Prediction (Bhavishyavani)
**Goal:** Restaurant ko kitne stars milenge, ye pehle se batana.

- **Preprocessing:** Computer ko sirf numbers samajh aate hain. Humne "Yes/No" (jaise 'Table Booking') ko 1 aur 0 mein badla.
- **Algorithm:** Humne **Decision Tree** use kiya. Ye ek "Flowchart" ki tarah kaam karta hai (e.g., "Agar votes 100 se zyada hain aur price kam hai, toh rating achhi hogi").
- **Key Insight:** Humne dekha ki **'Votes'** (kitne logon ne review diya) rating par sabse zyada asar dalte hain.

---

## 🍱 4. Task 2: Cuisine Classification (Vargikaran)
**Goal:** Restaurant ke data ko dekh kar ye batana ki wahan kaunsa khaana (Cuisine) milta hai.

- **Primary Cuisine:** Ek restaurant mein kai tarah ka khana ho sakta hai, par humne predict karne ke liye pehli wali cuisine ko main maana.
- **Algorithm:** Humne **Random Forest** use kiya. Ye bahut saare Decision Trees ka ek group hota hai, jo milkar faisla lete hain.
- **Challenge:** North Indian food ka data bahut zyada tha, isliye model usse jaldi pehchan leta hai par doosri cheezon mein kabhi kabhi galti karta hai.

---

## 🍕 5. Task 3: Recommendation System (Sujhav)
**Goal:** Agar aapko ek restaurant pasand hai, toh waisa hi doosra dhoondna.

- **TF-IDF:** Ye ek technique hai jo dekhti hai ki kaunsa word (cuisine name) kitna important hai.
- **Cosine Similarity:** Ye math ka use karke measure karta hai ki do restaurants ek doosre se kitne "mel" khate hain.
- **Result:** Agar aap "Italian" aur "Price Range 3" maangenge, toh computer pure dataset mein sabse similar restaurants dhoond kar nikal dega.

---

## 📍 6. Task 4: Location-based Analysis (Naksha Baazi)
**Goal:** Ye dekhna ki restaurants puri duniya ya shehar mein kahan kahan hain.

- **Visualization:** Humne Latitude aur Longitude ka use karke maps banaye. 
- **Stats:** Humne dekha ki London aur Orlando jaise sheharon mein average rating sabse achhi hai.
- **Pattern:** Humne paaya ki jitna mehnga restaurant hota hai, uski rating aksar utni hi thodi zyada hoti hai.

---

## 💻 7. Kaise Run Karein?
Aapko bas terminal mein ek command likhni hai:
```bash
python3 main.py
```
Iske baad aap ek menu dekh paayenge aur kisi bhi task ko select kar sakte hain.

---

**Machine Learning ek jaadu nahi hai, ye sirf purane data se pattern pehchanne ka ek tarika hai!**
Umeed hai aapko ye project samajhne mein maza aaya hoga.
