const initRDKitModule = require("@rdkit/rdkit");
initRDKitModule().then((RDKit) => {
  const mol = RDKit.get_mol("Fc1ccc(-c2cncc(CN3CCN(c4ccccc4F)CC3)c2)cc1");
  const fpText = mol.get_morgan_fp(JSON.stringify({ radius: 2, nBits: 2048 }));
  console.log("JS   fp sum:", fpText.split("").filter(x => x === "1").length);
  console.log("JS   fp head:", fpText.substring(0, 100));
});
