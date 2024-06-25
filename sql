with

raw as (
    select
        try_cast(PolicyIssuedDate as date) as policy_issued_date,
        try_cast(SUBMISSION_IDENTIFIER as enum('string')) as sub__identifier,
        case 
            when SUBMISSION_NEW_BUSINESS_INDICATO = 'Yes' then 1
            else 0
        end as sub__is_new_business,
        EXPERIAN_BIN as experian__bin,
        try_cast(SUBMISSION_QUOTE_NUMBER as integer) as sub__quote_numb,
        try_cast(QUOTE_VERSION_NUMBER as integer) as quote_version_numb,
        case
            when Unit='CCCC' then 0
            when Unit='Small Business' then 1
            when Unit='Commercial Lines' then 1
            else 0
        end as is_unit_small_business,
        
        {{recode_company_code_to_company_numb(SUBMISSION_COMPANY_CODE)}} as sub__company_numb,
        SUBMISSION_POLICY_SYMBOL as sub__policy_symbol,
        try_cast(SUBMISSION_POLICY_NUMBER as integer) as sub__policy_numb,
        
        Cinergy_PolicySymbol as cinergy__policy_symbol,
        try_cast(Cinergy_PolicyNumber as integer) as cinergy__policy_numb,
        try_cast(Cinergy_PolicyMod as integer) as cinergy__policy_mod,

        -- SUBMISSION_POLICY_EFFECTIVE_DATE,
        try_cast(SUBMISSION_POLICY_EFFECTIVE_DATE as date) as sub__policy_eff_date,
        -- SUBMISSION_POLICY_EXPIRATION_DAT,
        try_cast(SUBMISSION_POLICY_EXPIRATION_DATE as date) as sub__policy_exp_date,
        -- SUBMISSION_POLICY_STATUS_DESCRIP,
        SUBMISSION_POLICY_STATUS_DESCRIP as sub__policy_status_desc,
        -- SUBMISSION_POLICY_SUB_STATUS_DES,
        SUBMISSION_POLICY_SUB_STATUS_DES as sub__policy_substatus_desc,
        -- CIC_PREMIUM_AMOUNT,
        try_cast(CIC_PREMIUM_AMOUNT as float) as cic__premium,
        
        -- SUBMISSION_RECEIVED_DATE,
        try_cast(SUBMISSION_RECEIVED_DATE as date) as sub__received_date,
        -- SUBMISSION_CREATED_DATETIME,
        try_cast(SUBMISSION_CREATED_DATETIME as datetime) as sub__created_datetime,
        -- EXPERIAN_PRIMARY_NAICS_CODE,
        EXPERIAN_PRIMARY_NAICS_CODE as experian__primary_naics_code,
        -- EXPERIAN_SECOND_NAICS_CODE,
        EXPERIAN_SECOND_NAICS_CODE as experian__second_naics_code,
        -- EXPERIAN_THIRD_NAICS_CODE,
        EXPERIAN_THIRD_NAICS_CODE as experian__third_naics_code,
        -- EXPERIAN_FOURTH_NAICS_CODE,
        EXPERIAN_FOURTH_NAICS_CODE as experian__fourth_naics_code,
        -- FIRST_INSURED_NAICS_CODE,
        FIRST_INSURED_NAICS_CODE as first_insured__naics_code,
        -- NAICS_DESC,
        NAICS_DESC as naics_desc,
        -- NAICS_PREFIX_2_DESC,
        NAICS_PREFIX_2_DESC as naics_2_desc,
        -- NAICS_PREFIX_3_DESC,
        NAICS_PREFIX_3_DESC as naics_3_desc,
        -- FIRST_INSURED_NAICS_SOURCE_DESCR,
        FIRST_INSURED_NAICS_SOURCE_DESCR as first_insured__naics_source_desc,
        -- Product,
        Product as product,
        -- Program,
        Program as program,
        -- Agency_Owner_Name,
        Agency_Owner_Name as agy__owner_name,
        -- AGENCY_NUMBER,
        AGENCY_NUMBER as agy__numb,
        -- Agency_Name,
        Agency_Name as agy__name,
        -- Agency_Name_Nbr,
        Agency_Name_Nbr as agy__name_numb,
        -- CLD_Territory,
        CLD_Territory as cld_terr,
        -- AGCY_STATE_NM,
        AGCY_STATE_NM as agy__state_name,
        -- SALES_FIELD_REP_USER_ID,
        SALES_FIELD_REP_USER_ID as sales_field_rep__user_id,
        -- SALES_FIELD_REP_FULL_NAME,
        SALES_FIELD_REP_FULL_NAME as sales_field_rep__full_name,
        -- SALES_FIELD_REP_EMAIL,
        SALES_FIELD_REP_EMAIL as sales_field_rep__email,
        -- CURR_SALES_FIELD_REP_FULL_NAME,
        CURR_SALES_FIELD_REP_FULL_NAME as curr_sales_field_rep__full_name,
        -- CURR_HOME_OFF_REP_FULL_NM,
        CURR_HOME_OFF_REP_FULL_NM as curr_home_office_rep__full_name,
        -- HQ_REP_PREF_FULL_NM,
        HQ_REP_PREF_FULL_NM as hq_rep__pref_full_name,
        -- HOME_OFF_REP_USER_ID,
        HOME_OFF_REP_USER_ID as home_office_rep__user_id,
        -- HOME_OFF_REP_FULL_NM,
        HOME_OFF_REP_FULL_NM as home_office_rep__full_name,
        -- HOME_OFF_REP_EMAIL,
        HOME_OFF_REP_EMAIL as home_office_rep__email,
        -- CLIENT_IDENTIFIER,
        CLIENT_IDENTIFIER as client_id,
        -- FIRST_INSURED_FULL_NAME,
        FIRST_INSURED_FULL_NAME as first_insured__full_name,
        -- FIRST_INSURED_STREET_NAME,
        FIRST_INSURED_STREET_NAME as first_insured__street,
        -- FIRST_INSURED_CITY_NAME,
        FIRST_INSURED_CITY_NAME as first_insured__city,
        -- FIRST_INSURED_STATE_CODE,
        FIRST_INSURED_STATE_CODE as first_insured__state,
        -- FIRST_INSURED_ZIP_PLUS_FOUR_CODE,
        FIRST_INSURED_ZIP_PLUS_FOUR_CODE as first_insured__zip_plus_four,
        -- FIRST_INSURED_WEBSITE_ADDRESS,
        FIRST_INSURED_WEBSITE_ADDRESS as first_insured__website,
        -- FIRST_INSURED_ZIP_CODE,
        FIRST_INSURED_ZIP_CODE as first_insured__zip,
        -- FIRST_INSURED_ADDRESS_FIRST_LINE,
        FIRST_INSURED_ADDRESS_FIRST_LINE as first_insured__address_first_line,
        -- FIRST_INSURED_ADDRESS_SECOND_LIN,
        FIRST_INSURED_ADDRESS_SECOND_LIN as first_insured__address_second_line,
        -- SUBMISSION_COMMENT_CREATOR_USERN,
        SUBMISSION_COMMENT_CREATOR_USERN as sub__comment_creator_username,
        -- SUBMISSION_UPDATED_BY_USERNAME,
        SUBMISSION_UPDATED_BY_USERNAME as sub__updated_by_username,
        -- STREET_NAME,
        STREET_NAME as street,
        -- CITY_NAME,
        CITY_NAME as city,
        -- STATE_CODE,
        STATE_CODE as state,
        -- COUNTY_NAME,
        COUNTY_NAME as county,
        -- ZIP_CODE,
        ZIP_CODE as zip,
        -- ZIP_EXTENSION_CODE,
        ZIP_EXTENSION_CODE as zip_extension,
        -- ZIP_PLUS_FOUR_CODE,
        ZIP_PLUS_FOUR_CODE as zip_plus_four,
        -- AccountBusinessCategory,
        AccountBusinessCategory as acct__business_category,
        -- AccountBusinessService,
        AccountBusinessService as acct__business_service,
        -- AccountClassificationCode,
        AccountClassificationCode as acct__class_code,
        -- AccountClassificationDesc,
        AccountClassificationDesc as acct__class_desc,
        -- AccountNAICSCode,
        AccountNAICSCode as acct__naics_code,
        -- PILOT_AGENCY,
        case    
            when PILOT_AGENCY='Yes' then 1
            else 0
        end as is_pilot_agy,
        -- evolve_Premium_Amount,
        try_cast(evolve_Premium_Amount as double) as evolve__premium,
        -- Bound_Status,
        Bound_Status as bound_status,
        -- eVolve_Score,
        try_cast(eVolve_Score as double) as evolve__score,
        -- eVolve_Tier,
        eVolve_Tier as evolve__tier,
        -- eVolve_Tier_Intelliscore,
        eVolve_Tier_Intelliscore as evolve__tier_intelliscore,
        -- Experian_Intelliscore_214,
        Experian_Intelliscore_214 as experian__intelliscore_214,
        -- eVolve_EPPID,
        eVolve_EPPID as evolve__eppid,
        -- eVolve_IRPM,
        eVolve_IRPM as evolve__irpm,
        -- eVolve_Tier_YearsInBusiness,
        eVolve_Tier_YearsInBusiness as evolve__tier_years_in_business,
        -- eVolve_Company,
        {{recode_company_code_to_company_numb(eVolve_Company)}} as evolve__company_numb,
        case    
            when eVolve_PolicyTerm='3 Year' then '3 Years'
            else eVolve_PolicyTerm
        end as evolve__policy_term,
        try_cast(eVolve_QuoteNumber as integer) as evolve__quote_numb,
        -- Cinergy_AgencyContact,
        Cinergy_AgencyContact as cinergy__agency_contact,
        -- Cinergy_AgencyContact_Email,
        Cinergy_AgencyContact_Email as cinergy__agency_contact_email,
        -- Cinergy_AgencyContact_PhoneNbr,
        Cinergy_AgencyContact_PhoneNbr as cinergy__agency_contact_phone,
        -- Cinergy_Producer,
        Cinergy_Producer as cinergy__producer,
        
        -- EXPERIAN_YEARS_IN_FILE,
        EXPERIAN_YEARS_IN_FILE as experian__years_in_file,
        -- EXPERIAN_YEARS_IN_BUSINESS_CODE,\
        EXPERIAN_YEARS_IN_BUSINESS_CODE as experian__years_in_business_code,
        -- EXPERIAN_YEAR_BUSINESS_STARTED,
        EXPERIAN_YEAR_BUSINESS_STARTED as experian__year_business_started,
        -- EXPERIAN_EST_NBR_OF_EMPLOYEES,
        EXPERIAN_EST_NBR_OF_EMPLOYEES as experian__est_nbr_of_employees,
        -- EXPERIAN_EST_ANNUAL_SALES_AMT,
        EXPERIAN_EST_ANNUAL_SALES_AMT as experian__est_annual_sales_amount,

        try_cast(TRANSACTION_PROCESSED_DATETIME as timestamp) as transaction_processed_datetime,
        -- eVolve_IssuedDate,
        try_cast(eVolve_IssuedDate as date) as evolve__issued_date,
        -- Cinergy_PolicyCancelDate,
        try_cast(Cinergy_PolicyCancelDate as date) as cinergy__policy_cancel_date,
        -- eVolve_TransactionStatus,
        eVolve_TransactionStatus as evolve__transaction_status,
        -- eVolve_TransactionStatusPrior,
        eVolve_TransactionStatusPrior as evolve__transaction_status_prior,
        -- eVolve_PolicyState,
        eVolve_PolicyState as evolve__policy_state,
        -- Cinergy_StateCount,
        Cinergy_StateCount as cinergy__state_count,
        -- eVolve_VehicleCount,
        eVolve_VehicleCount as evolve__vehicle_count,
        -- AutoPolicy_Premium,
        try_cast(AutoPolicy_Premium as double) as auto_policy__premium,
        -- eVolve_HNO_Only,
        case 
            when eVolve_HNO_Only='Yes' then 1
            else 0
        end as evolve__hno_only,
        -- Cinergy_PropertyPremium,
        try_cast(Cinergy_PropertyPremium as double) as cinergy__property_premium,
        -- Cinergy_LiabilityPremium,
        try_cast(Cinergy_LiabilityPremium as double) as cinergy__liability_premium,
        -- Cinergy_TotalPropertyPrem,
        try_cast(Cinergy_TotalPropertyPrem as double) as cinergy__total_property_premium,
        -- Cinergy_BPP_Premium,
        try_cast(Cinergy_BPP_Premium as double) as cinergy__bpp_premium,
        -- Cinergy_Building_Premium,
        try_cast(Cinergy_Building_Premium as double) as cinergy__building_premium,
        -- Cinergy_ProfLiab_Prem,
        try_cast(Cinergy_ProfLiab_Prem as double) as cinergy__prof_liab_premium,
        -- Cinergy_TerrorismPrem,
        try_cast(Cinergy_TerrorismPrem as double) as cinergy__terrorism_premium,
        -- Cinergy_AtMMP,
        Cinergy_AtMMP as cinergy__at_mmp,
        
        -- Cinergy_EQ_Building_Limit,
        Cinergy_EQ_Building_Limit as cinergy__eq_building__limit,
        -- Cinergy_EQ_Building_Premium,
        Cinergy_EQ_Building_Premium as cinergy__eq_building__premium,
        
        -- Cinergy_EQ_BPP_Limit,
        Cinergy_EQ_BPP_Limit as cinergy__eq_bpp__limit,
        -- Cinergy_EQ_BPP_Premium,
        Cinergy_EQ_BPP_Premium as cinergy__eq_bpp__premium,
        -- Cinergy_DataDefender_Premium,
        Cinergy_DataDefender_Premium as cinergy__data_defender__premium,
        -- Cinergy_BOPNetworkDefender_Prem,
        Cinergy_BOPNetworkDefender_Prem as cinergy__network_defender__premium,
        -- Cinergy_HNOLiab_Premium,
        Cinergy_HNOLiab_Premium as cinergy__hno_liab__premium,
        -- Cinergy_BuildingIndicator,
        try_cast(Cinergy_BuildingIndicator as integer) as cinergy__has_building,
        -- Cinergy_BPPIndicator,
        try_cast(Cinergy_BPPIndicator as integer) as cinergy__has_bpp,
        -- Cinergy_ViewTopic,
        Cinergy_ViewTopic as cinergy__view_topic,


        -- ML_STP_Indicator,
        case 
            when ML_STP_Indicator='Yes' then 1
            else 0
        end as is_ml_stp,
        -- ML_Referral_Status,
        ML_Referral_Status as ml__referral_status,
        -- NonAgentToAgentPremiumRatio,
        NonAgentToAgentPremiumRatio as non_agent_to_agent_premium_ratio,
        -- MinutesBetweenInit_FirstRate,
        MinutesBetweenInit_FirstRate as minutes_between_init_first_rate,
        
        -- Cinergy_BlockedStatus,
        Cinergy_BlockedStatus as cinergy__blocked_status,
        -- Cinergy_BlockedReason,
        Cinergy_BlockedReason as cinergy__blocked_reason,
        -- Cinergy_BOP_PolicyStatus,
        Cinergy_BOP_PolicyStatus as cinergy__bop_policy_status,
        -- Cinergy_Auto_PolicyStatus,
        Cinergy_Auto_PolicyStatus as cinergy__auto_policy_status,
        -- Cinergy_WC_PolicyStatus,
        Cinergy_WC_PolicyStatus as cinergy__wc_policy_status,
        -- Cinergy_Umb_PolicyStatus,
        Cinergy_Umb_PolicyStatus as cinergy__umb_policy_status,
        -- Cinergy_PropertyBundleIndicator,
        case
            when Cinergy_PropertyBundleIndicator='Yes' then 1
            else 0
        end as cinergy__is_property_bundle,
        -- Cinergy_BOP_ISOTerritory,
        Cinergy_BOP_ISOTerritory as cinergy__bop_iso_terr,
        -- Cinergy_BOP_ISOTerritory_Desc,
        Cinergy_BOP_ISOTerritory_Desc as cinergy__bop_iso_terr_desc,
        -- Cinergy_PropertyCSPTerritory,
        Cinergy_PropertyCSPTerritory as cinergy__property_csp_terr,
        -- Cinergy_ProtectionClass,
        Cinergy_ProtectionClass as cinergy__protection_class,
        -- Cinergy_ConstructionType,
        Cinergy_ConstructionType as cinergy__construction_type,
        -- Cinergy_UmbrellaLimit,
        try_cast(Cinergy_UmbrellaLimit as double) / 1000000 as cinergy__umbrella_limit_$M,
        -- Cinergy_PropertyTIV,
        try_cast(Cinergy_PropertyTIV as double) as cinergy__property_tiv,
        -- Cinergy_DataDefender_Ind,
        case
            when Cinergy_DataDefender_Ind='Yes' then 1
            else 0
        end as cinergy__has_data_defender,
        -- Cinergy_NetworkDefender_Ind,
        case
            when Cinergy_NetworkDefender_Ind='Yes' then 1
            else 0
        end as cinergy__has_network_defender,

        -- Acct_FirstNamedInsured,
        Acct_FirstNamedInsured as acct__first_named_insured,
        -- Acct_BOPClassCode,
        Acct_BOPClassCode as acct__bop_class_code,
        -- Acct_BOPClassDescription,
        Acct_BOPClassDescription as acct__bop_class_desc,
        -- Acct_BOPClassIndustryGroup,
        Acct_BOPClassIndustryGroup as acct__bop_class_industry_group,
        -- Acct_BOPClassIndustrySubgroup,
        Acct_BOPClassIndustrySubgroup as acct__bop_class_industry_subgroup,
        -- Acct_YearsInBusiness,
        Acct_YearsInBusiness as acct__years_in_business,
        -- Acct_AnnualRevenue,
        try_cast(Acct_AnnualRevenue as double) as acct__annual_revenue,
        -- Acct_NumberOfFullTimeEmployees,
        try_cast(Acct_NumberOfFullTimeEmployees as integer) as acct__n_fulltime_employees,
        -- Acct_NumberOfPartTimeEmployees,
        try_cast(Acct_NumberOfPartTimeEmployees as integer) as acct__n_parttime_employees,
        -- Acct_TotalNumberOfEmployees,
        try_cast(Acct_TotalNumberOfEmployees as integer) as acct__n_total_employees,
        -- Acct_PriorCarrier,
        Acct_PriorCarrier as acct__prior_carrier,
        -- Acct_PriorCarrierCompanyID,
        try_cast(Acct_PriorCarrierCompanyID as integer) as acct__prior_carrier_company_id,
        -- Acct_AgencyExistCustomerQuestion,
        Acct_AgencyExistCustomerQuestion as acct__agency_exist_customer_question,
        -- Acct_PriorCarrierCompanyCIC,
        case
            when Acct_PriorCarrierCompanyCIC = 'True' then 1
            else 0
        end as acct__prior_carrier_company_cic,
        
        -- Cinergy_PolicyStatus,
        Cinergy_PolicyStatus as cinergy__policy_status,
        -- Cinergy_LegalEntity,
        Cinergy_LegalEntity as cinergy__legal_entity,
        -- Cinergy_BOP_HNO_CovExistsInd,
        case
            when Cinergy_BOP_HNO_CovExistsInd='True' then 1
            else 0
        end as cinergy__has_hno_cov,
        -- Cinergy_BOP_HNOLiab_Premium,
        try_cast(Cinergy_BOP_HNOLiab_Premium as double) as cinergy__hno_liab_premium,
        -- Building_Premium,
        try_cast(Building_Premium as double) as building__premium,
        -- TotalSquareFootage,
        try_cast(TotalSquareFootage as double) as total_square_footage,
        -- BuildingManualPremPerSqFt,
        try_cast(BuildingManualPremPerSqFt as double) as building__manual_prem_per_sqft,
        -- PolicyMinimumPremiumIndicator,
        case
            when PolicyMinimumPremiumIndicator='True' then 1
            else 0
        end as policy__is_min_premium,
        -- RatedSuccessfully,
        try_cast(RatedSuccessfully as integer) as is_rated_successfully,
        -- Rated,
        case
            when Rated='True' then 1
            else 0
        end as is_rated,

        -- Full_Policy_Number,
        Full_Policy_Number as full_policy_numb,

        -- Submission_Received_Month,
        Submission_Received_Month as sub__received_month,
        -- Submission_Received_Quarter,
        Submission_Received_Quarter as sub__received_quarter,
        -- Submission_Received_Year,
        Submission_Received_Year as sub__received_year,
        -- Submission_Received_Month_Name,
        Submission_Received_Month_Name as sub__received_month_name,
        
        -- Policy_Effective_Month,
        Policy_Effective_Month as policy_eff_month,
        -- Policy_Effective_Quarter,
        Policy_Effective_Quarter as policy_eff_quarter,
        -- Policy_Effective_Year,
        Policy_Effective_Year as policy_eff_year,
        -- Policy_Effective_Month_Name,
        Policy_Effective_Month_Name as policy_eff_month_name,
        
        -- Account_Date,
        Account_Date as acct__date,
        -- Account_Month,
        Account_Month as acct__month,
        -- Account_Quarter,
        Account_Quarter as acct__quarter,
        -- Account_Year,
        Account_Year as acct__year,
        
        -- QuoteDays,
        try_cast(QuoteDays as integer) as quote_days,
        -- QuoteAge,
        QuoteAge as quote_age,
        -- Cinergy_ReceivedThisWeek,
        case
            when Cinergy_ReceivedThisWeek='Yes' then 1
            else 0
        end as cinergy__is_received_this_week,
        -- Cinergy_ReceivedThisMonth,
        case
            when Cinergy_ReceivedThisMonth='Yes' then 1
            else 0
        end as cinergy__is_received_this_month,
        -- Cinergy_ReceivedThisYear,
        case 
            when Cinergy_ReceivedThisYear='Yes' then 1
            else 0
        end as cinergy__is_received_this_year,
        -- Cinergy_ReceivedLast7Days,
        case
            when Cinergy_ReceivedLast7Days='Yes' then 1
            else 0
        end as cinergy__is_received_last_7_days,
        -- Cinergy_ExpiringNext7Days,
        case
            when Cinergy_ExpiringNext7Days='Yes' then 1
            else 0
        end as cinergy__is_expiring_next_7_days,
        -- Policy_Premium,
        try_cast(Policy_Premium as double) as policy__premium,
        -- Evolve_Issued_Premium_Amount,
        try_cast(Evolve_Issued_Premium_Amount as double) as evolve__issued_premium,
        -- Premium_Band,
        Premium_Band as premium_band,
        -- NAICS_Sector,
        NAICS_Sector as naics_sector,
        -- Business_Segment,
        Business_Segment as business_segment,
        -- NAICS_Description,
        NAICS_Description as naics_desc,
        -- Cinergy_Quoted_Premium,
        try_cast(Cinergy_Quoted_Premium as double) as cinergy__quoted_premium,
        -- eVolve_Days_to_Issue,
        try_cast(eVolve_Days_to_Issue as integer) as evolve__days_to_issue,
        -- eVolve_Hit_Count,
        try_cast(eVolve_Hit_Count as integer) as evolve__hit_count,
        -- eVolve_Quote_Count,
        try_cast(eVolve_Quote_Count as integer) as evolve__quote_count,
        -- eVolve_BindableQuote_Count,
        try_cast(eVolve_BindableQuote_Count as integer) as evolve__bindable_quote_count,
        -- eVolve_AdvancedQuoteDays,
        try_cast(
            case
                when eVolve_AdvancedQuoteDays is null then 0
                else 
            end
            as integer) as evolve__advanced_quote_days
        -- eVolve_ManualPremium,
        try_cast(eVolve_ManualPremium as double) as evolve__manual_premium,
        -- UW_Uninvolved,
        try_cast(UW_Uninvolved as integer) as is_uw_uninvolved,
        -- STP_Count,
        try_cast(STP_Count as integer) as stp_count,
        -- Experian_BIN_Count,
        try_cast(Experian_BIN_Count as integer) as experian__bin_count,
        -- Source_QuoteNumber,
        try_cast(Source_QuoteNumber as integer) as source__quote_numb,
        -- Customer_Funnel_Status,
        Customer_Funnel_Status as customer_funnel_status,
        -- Individual_Awards,
        case 
            when Individual_Awards='Yes' then 1
            else 0
        end as has_individual_awards,
        -- Group_Awards,
        case
            when Group_Awards='Yes' then 1
            else 0
        end as has_group_awards,
        -- State_Submission_Day,
        try_cast(State_Submission_Day as integer) as state_submission_day,
        -- State_Issue_Day,
        try_cast(State_Issue_Day as integer) as state_issue_day,
        -- State_Effective_Day,
        try_cast(State_Effective_Day as integer) as state_effective_day,
        -- Cinergy_IncentiveProgram,
        case
            when Cinergy_IncentiveProgram='Post-Incentive Program' then 1
            else 0
        end as cinergy__is_post_incentive_program,
        case
            when Cinergy_IncentiveProgram='Pre-Incentive Program' then 1
            else 0
        end as cinergy__is_pre_incentive_program,
        -- Cinergy_Quoted_Tier,
        Cinergy_Quoted_Tier as cinergy__quoted_tier,
        -- Cinergy_Tier_Count,
        try_cast(Cinergy_Tier_Count as integer) as cinergy__tier_count,
        -- Cinergy_Tier_Group,
        Cinergy_Tier_Group as cinergy__tier_group,
        -- LRO_Indicator,
        case
            when LRO_Indicator='True' then 1
            else 0
        end as is_lro,
        -- Occupancy_Indicator,
        Occupancy_Indicator as occupancy_ind,
        -- Occupancy_Description,
        Occupancy_Description as occupancy_desc,
        -- EffectivePeriod,
        EffectivePeriod as effective_period,
        case
            when EffectivePeriod='After today' then 1
            else 0
        end as is_effective_after_today,
        -- EffectivePeriod2,
        EffectivePeriod2 as effective_period_2,
        -- LocationNumber,
        try_cast(LocationNumber as integer) as location_numb,
        -- location_AddressLineOne,
        location_AddressLineOne as location__address_line_one,
        -- location_AddressLineTwo,
        location_AddressLineTwo as location__address_line_two,
        -- location_CityName,
        location_CityName as location__city,
        -- location_State,
        location_State as location__state,
        -- location_ZipCode,
        location_ZipCode as location__zip,
        -- location_County,
        location_County as location__county,
        -- UWMethodCD,
        UWMethodCD as uw_method_cd,
        -- Cinergy_UserID,
        Cinergy_UserID as cinergy__user_id,
        -- ComparativeRaterVendor,
        ComparativeRaterVendor as comparative_rater_vendor,
        -- Cinergy_BIN,
        Cinergy_BIN as cinergy__bin,
        -- Cinergy_NumberOfEmployees,
        try_cast(Cinergy_NumberOfEmployees as integer) as cinergy__n_employees,
        -- State_BOP_ISOTerritory_Desc,
        State_BOP_ISOTerritory_Desc as state__bop_iso_terr_desc,
        -- TotalClassCode_Policy,
        TotalClassCode_Policy as n_class_code_policy,
        -- BookRollIndicator,
        case    
            when BookRollIndicator='Yes' then 1
            else 0
        end as is_book_roll,
        -- BOP_PrimaryClassCode,
        BOP_PrimaryClassCode as bop__primary_class_code,
        -- BOP_PrimaryClassCodeDesc,
        BOP_PrimaryClassCodeDesc as bop__primary_class_desc,
        -- BOP_PrimaryBusinessCategory,
        BOP_PrimaryBusinessCategory as bop__primary_business_category,
        -- BOP_PrimaryBusinessService,
        BOP_PrimaryBusinessService as bop__primary_business_service,
        -- BOP_PrimaryNAICS,
        BOP_PrimaryNAICS as bop__primary_naics,
        
        -- Cinergy_BusinessOwner_FullName,
        Cinergy_BusinessOwner_FullName as cinergy__business_owner_full_name,
        -- Cinergy_DBAName,
        Cinergy_DBAName as cinergy__dba_name,
        -- Cinergy_Distinct_BuildingCount,
        try_cast(Cinergy_Distinct_BuildingCount as integer) as cinergy__n_distinct_building,
        -- Mail_Loc_Address_Match,
        case
            when Mail_Loc_Address_Match='Yes' then 1
            else 0
        end as is_mail_loc_address_match,
        
        -- Cinergy_OriginalIssueDate,
        try_cast(Cinergy_OriginalIssueDate as date) as cinergy__original_issue_date,
        -- Cinergy_OriginalCreateDate,
        try_cast(Cinergy_OriginalCreateDate as date) as cinergy__original_create_date,
        -- Cinergy_CreateDate,
        try_cast(Cinergy_CreateDate as date) as cinergy__create_date,
        -- TestQuote_Indicator,
        try_cast(case
            when TestQuote_Indicator=1 then 1
            else 0
        end as integer) as is_test_quote,
        -- Cinergy_CreatedByUserName,
        Cinergy_CreatedByUserName as cinergy__created_by_username,
        -- Cinergy_CreatedByRoleName,
        Cinergy_CreatedByRoleName as cinergy__created_by_role_name,
        -- Cinergy_TransactionTypeXML
        Cinergy_TransactionTypeXML as cinergy__transaction_type_xml,
        -- Cinergy_TransactionStatusXML,
        Cinergy_TransactionStatusXML as cinergy__transaction_status_xml,
        -- Cinergy_PropInTheOpenIndicator,
        case
            when Cinergy_PropInTheOpenIndicator='True' then 1
            else 0
        end as cinergy__has_property_in_the_open,
        -- EachOccurrenceLimit,
        try_cast(try_cast(EachOccurrenceLimit as double)/1000000 as integer) as each_occurrence_limit_$M,
        -- AggregateLimit,
        try_cast(try_cast(AggregateLimit as double)/1000000 as integer) as aggregate_limit_$M,
        -- RentedToYouLimit,
        try_cast(try_cast(RentedToYouLimit as double)/1000000 as integer) as rented_to_you_limit_$M,
        -- PAndAInjuryLimit,
        try_cast(try_cast(PAndAInjuryLimit as double)/1000000 as integer) as p_and_a_injury_limit_$M,
        -- ProdAggregateLimit,
        try_cast(try_cast(ProdAggregateLimit as double)/1000000 as integer) as prod_aggregate_limit_$M,
        -- Issued_LastWeek,
        case
            when Issued_LastWeek='Yes' then 1
            else 0
        end as is_issued_last_week,
        -- UNDERWRITER_NAME,
        UNDERWRITER_NAME as underwriter
        -- ASSOCIATE_UNDERWRITER_NAME,
        ASSOCIATE_UNDERWRITER_NAME as associate_underwriter,
        -- TEAM_LEADER_NAME,
        TEAM_LEADER_NAME as team_leader,
        -- BUSINESS_UNIT_NAME,
        BUSINESS_UNIT_NAME as business_unit,
        -- LiabilityBasis_Count,
        try_cast(LiabilityBasis_Count as integer) as n_liability_bases,
        -- ProductSkippedIndicator,
        case 
            when ProductSkippedIndicator='Yes' then 1
            else 0
        end as is_product_skipped,
        -- Eval_Dt
        try_cast(Eval_Dt as date) as eval_date

    from {{source('hit_ratio_data_pipeline', 'tblquotes_hitratio_sb')}}


)

from raw
