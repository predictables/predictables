{%- macro recode_string_to_enum(column_name) -%}

{%- set output -%}
case
    when is_null({{ column_name }}) then 'null'
    else {{ column_name }}
end

{%- endset -%}

try_cast({{ output }} as enum('string'))
{%- endmacro -%}