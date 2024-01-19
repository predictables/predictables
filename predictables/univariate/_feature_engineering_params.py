def NON_NUMERIC_COLUMNS():
    out = [  # BIN numbers are not categories and not useful for modeling
        "experian_bin",
        "cinergy_bin"
        # these columns are not useful for modeling
        ,
        "cinergy_policy_numb",
        "cinergy_policy_module",
        "client_id",
        "evolve_eppid",
        "acct_prior_carrier_company_id"
        # if I manage to actually include 5-digit zip code, maybe I'll use this
        # (but I doubt it at this point)
        ,
        "zip_extension"
        # these columns are categorical integral codes
        ,
        "cinergy_bop_iso_territory",
        "cinergy_property_csp_territory"
        # most of these are also categorical integral codes
        ,
        "account_class_code",
        "acct_bop_class_code",
        "bop_primary_class_code",
        "cinergy_umbrella_limit",
        "experian_first_naics",
        "experian_second_naics",
        "experian_third_naics",
        "experian_primary_naics",
        "account_naics",
        "account_naics_2",
        "account_naics_3",
        "account_naics_4",
        "account_naics_5"
        # these columns are categorical codes
        ,
        "agy_numb"
        # these columns are repeated in the data
        ,
        "policy_prem",
        "cinergy_quoted_prem",
        "evolve_prem"
        # this one assumes what I am trying to predict -- there is only issued
        # premium if there is a hit
        ,
        "evolve_issued_prem",
    ]
    return out


def NON_CATEGORICAL_COLUMNS():
    out = [  # already include the codes for the following columns, don't need a description
        # as well (unless I make the vector embeddings for the codes and descriptions)
        "naics_desc",
        "naics_2_desc",
        "naics_3_desc",
        "account_sic_desc",
        "account_naics_desc",
        "account_naics_desc2",
        "account_naics_2_desc",
        "account_naics_3_desc",
        "account_naics_4_desc",
        "account_naics_5_desc"
        # these are the same as the naics2 and naics3 descriptions
        ,
        "naics_sector",
        "naics_description"
        # same idea, but for class code:
        ,
        "account_class_desc",
        "bop_primary_class_code_desc",
        "acct_bop_class_desc"
        # not sure about this one
        ,
        "submission_id"
        # policy symbols only really tell me the product -- I am already assuming that
        # the product is BOP
        ,
        "sub_policy_sym",
        "cinergy_policy_sym",
        "full_policy_number"
        # unclear whether or not I should include these, or whether they will be
        # known when the model is used
        ,
        "sub_policy_status_desc",
        "cinergy_bop_policy_status",
        "cinergy_auto_policy_status",
        "cinergy_wc_policy_status",
        "cinergy_umb_policy_status"
        # additional agency information, but having the full agency number is enough
        ,
        "agy_name",
        "agy_name_numb"
        # many different people -- names, phone numbers, emails, etc.
        # it is possible that there is an effect from the quality of the agent, but
        # I am not sure how to measure that at this point
        # TODO: should I include this -- is the quality of the agent/uw important?
        ,
        "SALES_FIELD_REP_USER_ID",
        "hq_rep_pref_full_name",
        "sub_comment_creator_username",
        "sub_updated_by_username"
        # address information -- I am already including zip code, which is the finest
        # level of granularity that I could see being useful (and is already a
        # stretch given the high dimensionality of zip code data)
        ,
        "first_insured_zip_plus_4",
        "zip_plus_4",
        "first_insured_zip",
        "zip",
        "location_zip",
        "street",
        "first_insured_city",
        "city",
        "location_city",
        "location_county",
        "source_quote_numb",
        "join",
        "cinergy_viewtopic",
        "acct_prior_carrier",
        "acct_prior_carrier_company_id",
    ]
    return out


def NUMERIC_CATEGORICAL_COLUMNS():
    out = [
        # territory codes
        "cinergy_bop_iso_territory",
        "cinergy_property_csp_territory"
        # coverage limits for umbrella (if also purchased)
        ,
        "cinergy_umbrella_limit",
        "account_class_code",
        "experian_first_naics",
        "experian_second_naics",
        "experian_third_naics",
        "experian_primary_naics",
        "account_naics_2",
        "account_naics_3",
        "account_naics_4",
        "account_naics_5",
        "cinergy_quoted_tier",
        "total_class_code_policy",
        "acct_prior_carrier_company_id",
    ]

    return out
