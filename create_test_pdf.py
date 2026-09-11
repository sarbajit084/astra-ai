import fitz

def make_sample_pdf():
    doc = fitz.open()
    page = doc.new_page()
    
    text = (
        "EMPLOYMENT OFFER LETTER\n\n"
        "Date: September 5, 2026\n"
        "To: Biswajit Manna\n"
        "Position: Senior AI Software Engineer\n"
        "Company: Apex Global Technologies Inc.\n\n"
        "Dear Biswajit,\n\n"
        "We are thrilled to extend an offer for the position of Senior AI Software Engineer at Apex Global Technologies. "
        "Your official start date is scheduled for October 1, 2026.\n\n"
        "COMPENSATION & BENEFITS:\n"
        "- Base Annual Salary: $145,000 USD paid bi-weekly.\n"
        "- Annual Performance Bonus: Target of 15% based on individual and corporate milestones.\n"
        "- Stock Options: 10,000 equity units vesting over a 4-year schedule with a 1-year cliff.\n"
        "- Health & Wellness: Comprehensive medical, dental, and vision insurance coverage effective on day 1.\n"
        "- Paid Time Off: 22 days of annual vacation plus 10 public holidays.\n"
        "- Remote Work Allowance: $1,500 home office setup stipend and $100 monthly internet reimbursement.\n\n"
        "TERMS & CONDITIONS:\n"
        "This offer is contingent upon successful completion of standard reference and background checks. "
        "Please sign and return this offer letter by September 15, 2026 to confirm your acceptance.\n\n"
        "Sincerely,\n"
        "Human Resources Department\nApex Global Technologies Inc."
    )
    
    rect = fitz.Rect(50, 50, 550, 750)
    page.insert_textbox(rect, text, fontsize=11, fontname="helv", align=0)
    doc.save("test_offer_letter.pdf")
    doc.close()
    print("test_offer_letter.pdf created successfully!")

if __name__ == "__main__":
    make_sample_pdf()
